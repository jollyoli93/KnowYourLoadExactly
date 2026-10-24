from pathlib import Path

import numpy as np
import torch
from joblib import load

from pathlib import Path
import numpy as np
import torch
from joblib import load

class MLP(torch.nn.Module):
    def __init__(self, in_feat, out_feat, hidden) -> None:
        super().__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_feat, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, out_feat)
        )

    def forward(self, x):
        x = self.linear(x)
        return x


class Roi_Predict:
    def __init__(self, model_path, scaler_dir=None):
        self.model = MLP(5, 2, 250)
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

        # Custom scaler
        if scaler_dir is None:
            scaler_dir = Path(__file__).resolve().parent.parent / "Models" / "scalars"
        else:
            scaler_dir = Path(scaler_dir)

        self.feat_scaler = load(scaler_dir / "feat_scaler.pkl")
        self.target_scaler = load(scaler_dir / "target_scaler.pkl")

    def predict_load_roi(self, img_size, hook_coords):
        img_height, img_width = int(img_size[0]), int(img_size[1])

        x, y, w, h = np.asarray(hook_coords, dtype=np.float32).reshape(-1)
        yb_hook = y + (h / 2.0)

        hook_features = np.array([[x, y, w, h, yb_hook]], dtype=np.float32)
        xs_scaled = self.feat_scaler.transform(hook_features)

        device = next(self.model.parameters()).device
        xs_t = torch.from_numpy(xs_scaled).float().to(device)
        
        with torch.no_grad():
            pred_scaled = self.model(xs_t).cpu().numpy()

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