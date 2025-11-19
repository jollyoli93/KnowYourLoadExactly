import torch
import albumentations as A
import numpy as np

class Load_Classify:
    def __init__(self, path_to_model="./Models/crane_models/crane_classifier.pth"):
        self.path = path_to_model
        checkpoint = torch.load(
            self.path,
            map_location='cpu',
            weights_only=False
        )

        self.class_model = checkpoint["model"]
        self.class_model.eval()

        if torch.cuda.is_available():
            self.class_model.cuda()

        self.cls_names = checkpoint["cls_names"]
        self.cls_index = checkpoint["cls_index"]
        self.val_transforms = A.Compose([
            A.Resize(height=240, width=240),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                        max_pixel_value=255.0),
        ])
        
    def predict_one_image(self, image_array):
        model = self.class_model
        transforms = self.val_transforms

        # model.eval()
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