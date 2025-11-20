import torch

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

class Crop_Predict:
    def __init__(self, model_path="app/Models/crane_models/LoadRegressionDims.pth"):
        self.model = LinearRegression(14, 2, 512)
        state = torch.load(model_path,
                           map_location="cpu",
                           weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

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

        pred = self.model(input_feats)
        img_dims = torch.tensor([img_width, img_height])
        xtens, ytens = pred * img_dims

        xc, yc = int(xtens), int(ytens)
        w, h = int(img_width * 0.07), int(img_height * 0.07)

        yt = yc - (h // 2)
        xt = xc - (w // 2)

        return xt, yt, w, h