from pathlib import Path

import numpy as np
import torch
from joblib import load

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

class Roi_Predict:
  def __init__(self, model_path):
    self.model = LinearRegression(5, 2, 64)
    state = torch.load(model_path,
                        map_location="cpu",
                        weights_only=True)
    self.model.load_state_dict(state)
    self.model.eval()

    BASE_DIR = Path(__file__).resolve().parent
    feat_scaler_path = BASE_DIR / "scalars" / "feat_scaler.pkl"
    targ_scaler_path = BASE_DIR / "scalars" / "target_scaler.pkl"

    self.feat_scalar = load(feat_scaler_path)
    self.target_scalar = load(targ_scaler_path)

  def predict_load_roi(self, img_size, hook_coords):
    """
    hook_coords: normalized YOLO (x_center, y_center, w, h) in [0,1]
    img_size: (img_height, img_width) or (H, W, C) - we use first two.
    dims: optional crop size control
    returns: x top, y top, roi width, roi height
    """

    # image dims
    img_height = int(img_size[0])
    img_width  = int(img_size[1])

    hook_coords_np = np.asarray(hook_coords, dtype=np.float32).reshape(-1)

    x, y, w, h = hook_coords_np  # normalized
    yb_hook = y + (h / 2.0)

    # features for scaler: [x, y, w, h, yb_hook]
    hook_features = np.array([[x, y, w, h, yb_hook]], dtype=np.float32)
    xs_scaled = self.feat_scalar.transform(hook_features)  # numpy (1, 5)

    # Get predictions
    xs_t = torch.from_numpy(xs_scaled).float() #.to(device)
    self.model.eval()
    with torch.no_grad():
        pred_scaled = self.model(xs_t).cpu().numpy()

      # pred_inv is [dx, dy]
    pred_inv = self.target_scalar.inverse_transform(pred_scaled.reshape(1, -1))
    print(pred_inv)
    # dx, dy = pred_inv[0]
    dx, dy = float(pred_inv[0, 0]), float(pred_inv[0, 1])

    # ---- apply offsets relative to hook size (still normalized [0-1]) ----
    x_pred = x + (dx * w)
    y_pred = yb_hook + (dy * h)

    xc = int(round(x_pred * img_width ))
    yc = int(round(y_pred * img_height))

    #crop to top of hook
    gap_pixels = yc - (yb_hook * img_height)

    crop_h = int(gap_pixels * 1.6)
    crop_w = int(crop_h * 1.2)      # Usually loads are wider than they are tall

    xt = int(xc - (crop_w // 2))
    yt = int(yc - (crop_h // 2))

    # Clamp to image bounds
    xt = max(0, min(xt, img_width - crop_w))
    yt = max(0, min(yt, img_height - crop_h))

    return xt, yt, crop_w, crop_h