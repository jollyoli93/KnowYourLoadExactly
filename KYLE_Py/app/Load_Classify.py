import threading
import albumentations as A
from pathlib import Path
import numpy as np
import onnxruntime as rt
from PIL import Image

from app.image_tools import image_processes

def is_valid_image(path):
    print(path, "Hello")
    if not isinstance(path, (str, Path)):
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError):
        return False


class Load_Classify:
    input_cls = [
        'background', 'beams', 'concrete', 'formwork',
        'pipes', 'plant', 'precast', 'rebar', 'skip'
    ]

    def __init__(self, path_to_model=None):
        # Fallback to default absolute path if none provided
        # self.path = path_to_model or DEFAULT_MODEL_PATH
        self.input_size = (240, 240)
        self.sess = None
        self.input_name = None
        self.output_name = None
        
        self.load_model_onnx()

    def load_model_onnx(self):
        self.sess = ModelCache.get_session(self.path)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def predict_one_image(self, input_img):
        try:
            if isinstance(input_img, np.ndarray):
                print("Not ready")

                # ImageNet stats
                # 240px - slightly larger than the 224px training size
                val_transforms = A.Compose([
                    A.Resize(height=240, width=240),
                    A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225),
                            max_pixel_value=255.0),
                    ])
                transforms = val_transforms
                transformed = transforms(image=input_img)['image']
                processed_img = np.transpose(transformed, (2, 0, 1))
                processed_img = processed_img[None, ...]
                print(processed_img.shape)

            elif is_valid_image(input_img):
                print("Valid")
                processed_img = image_processes(input_img, self.input_size)
            else:
                raise ValueError("Input is not a valid numpy array or readable image path.")
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None

        results = self.sess.run([self.output_name], {self.input_name: processed_img})[0]
        
        pred_idx = np.argmax(results)
        pred_class = self.input_cls[pred_idx]
        confidence = float(results.flat[pred_idx])
        
        return pred_class, confidence