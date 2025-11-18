# from wandb.integration.ultralytics import add_wandb_callback
from ultralytics import YOLO
import cv2
import numpy as np
import torch
import logging
import albumentations as A

logging.basicConfig(filename="crane_detect.log",
                    format='%(asctime)s %(message)s',
                    level=logging.INFO,
                    filemode='w')

logger = logging.getLogger()

best_weights = "./crane_models/yolov8_basic.pt"
yolo_model = YOLO(best_weights)

test_images = ['../images/2503061230060000_jpg.rf.25f5fb4afcbba2631a14691fc0869d85.jpg', '../images/2507190940060000_jpg.rf.25413f580b6dab2700fab296e0eca813.jpg']

# # Try one image
# image = 'images/2507190940060000_jpg.rf.25413f580b6dab2700fab296e0eca813.jpg'
# img = cv2.imread(image)  #1 hook, 1 load real

def cv_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    
# cv_show("Crane", img)

class LinearRegression(torch.nn.Module):
    def __init__(self, in_feat, out_feat, hidden) -> None:
        super().__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_feat, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, out_feat),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.linear(x)
        return x
    
cropPredict = LinearRegression(14, 2, 512)
cropPredict.load_state_dict(torch.load("./crane_models/LoadRegressionDims.pth", weights_only=True, map_location=torch.device('cpu')))
cropPredict.eval()

def predict_load_crop(hook_coords, img_size, dims):
  img_width = img_size[1]
  img_height = img_size[0]
  print(f"Predicting new coords")

  xn = dims[0]/img_width
  yn = dims[1]/img_height
  width_n = dims[2]/img_width
  height_n = dims[3]/img_height
  area = width_n*height_n
  ratio = width_n/height_n

  input = torch.concat((hook_coords, torch.tensor([width_n]), torch.tensor([height_n]), torch.tensor([area]), torch.tensor([ratio]),
                        torch.tensor([xn]), torch.tensor([yn])), dim=0) #Changed axis to dim from Colab

  predict = cropPredict(input) #x, y
  img_dims = torch.tensor([img_width, img_height])
  xtens,ytens = predict*img_dims

  xc, yc = int(xtens), int(ytens)

  # Toggle size of crop with respect to size of image
  w , h = int(img_width*0.07), int(img_height*0.07)

  #Get top left and top right x,y
  yt = yc - (h//2)
  xt = xc - (w//2)

  print(f"Crop dims: {xt, yt, w, h}")
  return xt, yt, w, h

  # return standard_size #Return top left and bottom right to crop
  
def crop_load(img, x, y, w, h):
  """
    Takes as input, top left corner, width and height.
  """
  crop_img = img.copy()
  cropped_img = crop_img[y:y+h,x:x+w]

  return cropped_img

checkpoint = torch.load('./crane_models/crane_classifier.pth', map_location='cpu', weights_only=False)

class_model = checkpoint['model']
class_model.eval()

if torch.cuda.is_available():
    class_model = class_model.cuda()

cls_names = checkpoint['cls_names']
cls_index = checkpoint['cls_index']

print("Model loaded successfully!")
logger.info("Model loaded successfully!")

val_transforms = A.Compose([
    A.Resize(height=240, width=240, p=1.0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
])

def predict_one_image(model, image_array, transforms=val_transforms):
    """
    Predict class for a single image from test set
    """
    model.eval()

    # Apply transforms
    transformed = transforms(image=image_array)['image']
    x = np.transpose(transformed, (2, 0, 1))
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    # Move to GPU if available
    if torch.cuda.is_available():
        x = x.cuda()

    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()

    pred_class = cls_names[pred_idx]

    return pred_class, confidence

# Run prediction on the image
# results = yolo_model.predict(img, stream=False)

# Run on multiple images
results = yolo_model.predict(test_images, stream=True)

# Iterate over the results
count = 0
for result in results:
    logger.info(f"Running image {count}")
    img = cv2.imread(test_images[count])
    count += 1
    
    boxes = result.obb.cpu()  # Get boxes on CPU in numpy format
    for box in boxes:  # Iterate over boxes - each box is a prediction (Multiple predictions will run each)
        logger.info("Start of box predict")
        # print(box, f" printing box obj")
        r = box.xyxy[0].numpy() .astype(int) # Get corner points as int (Actual Co-ords)
        class_id = int(box.cls[0])  # Get class ID
        class_name = yolo_model.names[class_id]  # Get class name using the class ID
        conf = float(box.conf[0])
        print(f"Class: {class_name}, Box: {r}")  # Print class name and box coordinates
        label = f"{class_name}: {conf:.2f}"

        # Get predicted co-ordinates if not load detected (Currently duplicated load crop, need to cancel it)

        if class_id == 0: #Hook Class - doesn't capture hook
          logger.info(f"No hook captured, running {class_id}")
          coords = box.xyxyxyxyn[0].flatten() #Get all coordinates for prediction
          print(f"XY n {coords}")
          dims = box.xywhr[0]

          # Predict location for crop
          pts = predict_load_crop(coords, box.orig_shape, dims)
          xt, yt, w, h = pts
          crop = crop_load(img, xt,yt,w,h)
          # crop_resized = cv2.resize(crop, (240, 240))

          # Show cropped region
        #   cv_show("crop", crop)
          # classify crop
          predicted_class, confidence = predict_one_image(class_model, crop)

          print(f"Predicted Cropped Class: {predicted_class}")
          print(f"Confidence of Crop: {confidence:.4f} ({confidence*100:.2f}%)")

          cv2.rectangle(img, (xt, yt), (xt + w, yt + h), (0, 255, 0), 2) # type: ignore
          cv2.putText(img, f"{predicted_class}: {confidence:.2f}", (xt, yt - 10), # type: ignore
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA) # type: ignore
          cv_show("Hook", img)

        if class_id == 1: #Load Class - doesn't capture hook
          logger.info(f"Load crop found, running {class_id}")
          xt, yt, xb, yb = r
          w, h = box.xywhr.numpy()[0].astype(int)[2:4] # type: ignore
          crop = crop_load(img, xt,yt,w,h)
          # crop_resized = cv2.resize(crop, (200, 200))
        #   cv_show("crop", crop)

          #classify
          predicted_class, confidence = predict_one_image(class_model, crop)
          print(f"Predicted Original Class: {predicted_class}")
          print(f"Confidence of Original: {confidence:.4f} ({confidence*100:.2f}%)")

          cv2.rectangle(img, r[:2], r[2:], (0, 255, 0), 2) # pyright: ignore[reportCallIssue, reportArgumentType]
          cv2.putText(img, f"{predicted_class}: {confidence:.2f}", (xt, yt - 10), # type: ignore
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA) # pyright: ignore[reportCallIssue]
          cv_show("Load", img)

        logger.info("End of box")
    logger.info("End of result")
logger.info("End.")
