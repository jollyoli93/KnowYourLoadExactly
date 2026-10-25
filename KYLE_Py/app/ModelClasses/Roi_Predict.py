from pathlib import Path
import onnxruntime as rt
import numpy as np
from joblib import load


class Roi_Predict:
    def __init__(self, session, scaler_dir=None):
        self.session = session

        # Custom scaler
        if scaler_dir is None:
            scaler_dir = Path(__file__).resolve().parent.parent / "Models" / "scalars"
        else:
            scaler_dir = Path(scaler_dir)

        self.feat_scaler = load(scaler_dir / "feat_scaler.pkl")
        self.target_scaler = load(scaler_dir / "target_scaler.pkl")

    def _get_predition(self, inputs):
        if self.session:
            onnxruntime_input = {input_arg.name: input_value for input_arg, input_value in zip(self.session.get_inputs(), inputs)}

            onnxruntime_outputs = self.session.run(None, onnxruntime_input)[0]
            return onnxruntime_outputs
        else:
            raise ValueError("Session can't be empty")
        
    def predict_load_roi(self, img_size, hook_coords):
        img_height, img_width = int(img_size[0]), int(img_size[1])

        x, y, w, h = np.asarray(hook_coords, dtype=np.float32).reshape(-1)
        yb_hook = y + (h / 2.0)

        hook_features = np.array([[x, y, w, h, yb_hook]], dtype=np.float32)
        xs_scaled = self.feat_scaler.transform(hook_features)
        print(xs_scaled)

        pred_scaled = self._get_predition(xs_scaled)
        
        pred_inv = self.target_scaler.inverse_transform(pred_scaled.reshape(1, -1))
        dx, dy = float(pred_inv[0, 0]), float(pred_inv[0, 1])

        x_offset, y_offset = dx * w, dy * h
        x_pred, y_pred = x + x_offset, yb_hook + y_offset

        xc = int(round(x_pred * img_width))
        yc = int(round(y_pred * img_height))

        hook_h_s = (h * img_height) / 100
        hook_w_s = (w * img_width) / 100

        crop_h = min(int(y_offset * img_height * (1 + hook_h_s)), int(y_offset * img_height * 1.6))  
        crop_w = int(crop_h * (1 + hook_w_s/10))     

        # check to ensure crop dimensions don't exceed image dimensions
        crop_h = min(crop_h, img_height)
        crop_w = min(crop_w, img_width)

        xt = int(xc - (crop_w // 2))
        yt = int(yc - (crop_h // 2))

        # Clamp starting coordinates
        xt = max(0, min(xt, img_width - crop_w))
        yt = max(0, min(yt, img_height - crop_h))

        return xt, yt, crop_w, crop_h