import torch

def to_rgb(img):
    if isinstance(img, torch.Tensor):
        if img.shape[0] == 1:
            img = img.repeat((3, 1, 1))
        if img.shape[0] == 4:
            img = img[:3]
    else:
        if img.mode == "L":
            img = img.convert('RGB')
    return img