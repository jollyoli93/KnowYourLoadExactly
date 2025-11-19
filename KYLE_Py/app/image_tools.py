import torch

def cv_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    
def predict_load_crop(hook_coords, img_size, dims, model):
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

    pred = model(input_feats)
    img_dims = torch.tensor([img_width, img_height])
    xtens, ytens = pred * img_dims

    xc, yc = int(xtens), int(ytens)
    w, h = int(img_width * 0.07), int(img_height * 0.07)

    yt = yc - (h // 2)
    xt = xc - (w // 2)

    return xt, yt, w, h