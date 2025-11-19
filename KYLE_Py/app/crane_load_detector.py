from ultralytics import YOLO
import cv2
import numpy as np
import torch
import albumentations as A
from pathlib import Path


def cv_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()


class LinearRegression(torch.nn.Module):
    def __init__(self, in_feat, out_feat, hidden):
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
        return self.linear(x)


class KYLE:

    def __init__(self, yolo_weights):

        # -------------------------------
        # YOLO
        # -------------------------------
        self.weights = yolo_weights
        self.yolo_model = YOLO(self.weights)

        # -------------------------------
        # Test images
        # -------------------------------
        # self.test_images = test_images

        # -------------------------------
        # Load Crop Regression Model
        # -------------------------------
        self.cropPredict = LinearRegression(14, 2, 512)
        state = torch.load("./crane_models/LoadRegressionDims.pth",
                           map_location="cpu",
                           weights_only=True)
        self.cropPredict.load_state_dict(state)
        self.cropPredict.eval()

        # -------------------------------
        # Load Classifier
        # -------------------------------
        checkpoint = torch.load(
            "./crane_models/crane_classifier.pth",
            map_location='cpu',
            weights_only=False
        )

        self.class_model = checkpoint["model"]
        self.class_model.eval()

        if torch.cuda.is_available():
            self.class_model.cuda()

        self.cls_names = checkpoint["cls_names"]
        self.cls_index = checkpoint["cls_index"]

        print("Model loaded successfully!")

        # -------------------------------
        # Albumentations transform
        # -------------------------------
        self.val_transforms = A.Compose([
            A.Resize(height=240, width=240),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                        max_pixel_value=255.0),
        ])

    # ----------------------------------------------------------------------
    # Prediction modules
    # ----------------------------------------------------------------------
    def predict_load_crop(self, hook_coords, img_size, dims):
        img_width = img_size[1]
        img_height = img_size[0]

        xn = dims[0] / img_width
        yn = dims[1] / img_height
        width_n = dims[2] / img_width
        height_n = dims[3] / img_height
        area = width_n * height_n
        ratio = width_n / height_n

        input_feats = torch.concat((
            hook_coords,
            torch.tensor([width_n, height_n, area, ratio, xn, yn])
        ), dim=0)

        pred = self.cropPredict(input_feats)
        img_dims = torch.tensor([img_width, img_height])
        xtens, ytens = pred * img_dims

        xc, yc = int(xtens), int(ytens)
        w, h = int(img_width * 0.07), int(img_height * 0.07)

        yt = yc - (h // 2)
        xt = xc - (w // 2)

        return xt, yt, w, h

    def crop_load(self, img, x, y, w, h):
        return img[y:y+h, x:x+w]

    # ----------------------------------------------------------------------
    # Classifier wrapper
    # ----------------------------------------------------------------------
    def predict_one_image(self, image_array):
        model = self.class_model
        transforms = self.val_transforms

        model.eval()
        transformed = transforms(image=image_array)['image']
        x = np.transpose(transformed, (2, 0, 1))
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

        if torch.cuda.is_available():
            x = x.cuda()

        with torch.no_grad():
            output = model(x)
            probs = torch.softmax(output, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item()

        pred_class = self.cls_names[pred_idx]
        return pred_class, confidence

    # ----------------------------------------------------------------------
    # YOLO detection
    # ----------------------------------------------------------------------
    def detect(self, results, image):
        for idx, result in enumerate(results):
            # img = cv2.imread(image[idx]) stream
            img = cv2.imread(image)
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

                    xt, yt, w, h = self.predict_load_crop(coords, box.orig_shape, dims)

                    crop = self.crop_load(img, xt, yt, w, h)
                    pred_class, confidence = self.predict_one_image(crop)

                    print(f"Predicted Crop: {pred_class}, Conf {confidence:.4f}")

                    cv2.rectangle(img, (xt, yt), (xt+w, yt+h), (0, 255, 0), 2)
                    cv2.putText(img, f"{pred_class}: {confidence:.2f}",
                                (xt, yt - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    cv_show("Hook", img)

                # -----------------------------------
                # LOAD (class 1)
                # -----------------------------------
                elif class_id == 1:
                    xt, yt, xb, yb = r
                    w, h = box.xywhr.numpy()[0].astype(int)[2:4]

                    crop = self.crop_load(img, xt, yt, w, h)
                    pred_class, confidence = self.predict_one_image(crop)

                    print(f"Pred Original: {pred_class}, Conf {confidence:.4f}")

                    cv2.rectangle(img, r[:2], r[2:], (0, 255, 0), 2)
                    cv2.putText(img, f"{pred_class}: {confidence:.2f}",
                                (xt, yt - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    cv_show("Load", img)

    # ----------------------------------------------------------------------
    # Public functions
    # ----------------------------------------------------------------------
    def detect_image(self, image):
        results = self.yolo_model.predict(image, stream=False)
        self.detect(results, image)

    def detect_stream(self, stream):
        results = self.yolo_model.predict(stream, stream=True)
        self.detect(results, stream)
