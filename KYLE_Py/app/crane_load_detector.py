from ultralytics import YOLO
import cv2
import torch
from . import image_tools
from pathlib import Path

from app.Models.Crop_Predict import Crop_Predict
from app.Models.Load_Classify import Load_Classify


def cv_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

class KYLE:

    def __init__(self, yolo_weights):
        # YOLO
        self.weights = yolo_weights
        self.yolo_model = YOLO(self.weights)
        
        # Load Crop finder Regression Model
        self.crop_predict = Crop_Predict() #remove hardcoded path
        # Load Classifier
        self.classify_model = Load_Classify()

    def crop_load(self, img, x, y, w, h):
        return img[y:y+h, x:x+w]

    # YOLO detection
    def detect(self, results, image):
      load_results = []
      
      for idx, result in enumerate(results):
          img = None
          if (type(image) is not list):
            img = cv2.imread(image)
          else: #is array(list) - issue may result if the stream isnt a list
            img = cv2.imread(image[idx])  
            
          image_results = {}  
          
          boxes = result.obb.cpu()
          for box in boxes:
              r = box.xyxy[0].numpy().astype(int)
              class_id = int(box.cls[0])
              class_name = self.yolo_model.names[class_id]
              conf = float(box.conf[0])

              print(f"Class: {class_name}, Box: {r}")

              # -----------------------------------
              # HOOK (class 0)
              # -----------------------------------
              if class_id == 0:
                  coords = box.xyxyxyxyn[0].flatten()
                  dims = box.xywhr[0]

                  xt, yt, w, h = self.crop_predict.predict_load_crop(coords, box.orig_shape, dims)

                  crop = self.crop_load(img, xt, yt, w, h)
                  pred_class, confidence = self.classify_model.predict_one_image(crop)
                  image_results[pred_class] = [confidence, xt, yt, w, h]
                  
                #   # DEBUG STEPS
                #   print(f"Predicted Crop: {pred_class}, Conf {confidence:.4f}")
                #   cv2.rectangle(img, (xt, yt), (xt+w, yt+h), (0, 255, 0), 2)
                #   cv2.putText(img, f"{pred_class}: {confidence:.2f}",
                #               (xt, yt - 5),
                #               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                #   cv_show("Hook", img)

              # -----------------------------------
              # LOAD (class 1)
              # -----------------------------------
              elif class_id == 1:
                  xt, yt, xb, yb = r
                  w, h = box.xywhr.numpy()[0].astype(int)[2:4]

                  crop = self.crop_load(img, xt, yt, w, h)
                  pred_class, confidence = self.classify_model.predict_one_image(crop)

                  image_results[pred_class] = [confidence, xt, yt, w, h]

                #   # DEBUG Steps
                #   print(f"Pred Original: {pred_class}, Conf {confidence:.4f}")
                #   cv2.rectangle(img, r[:2], r[2:], (0, 255, 0), 2)
                #   cv2.putText(img, f"{pred_class}: {confidence:.2f}",
                #               (xt, yt - 5),
                #               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                #   cv_show("Load", img)
                  
          load_results.append(image_results)
      return load_results

    #Run inference on one image
    def detect_image(self, image):
        yolo_results = self.yolo_model.predict(image, stream=False)
        crop_results = self.detect(yolo_results, image)
        print(crop_results)

    #Run inference on bulk images - upload array of images.
    def detect_stream(self, stream):
        results = self.yolo_model.predict(stream, stream=True)
        self.detect(results, stream)
