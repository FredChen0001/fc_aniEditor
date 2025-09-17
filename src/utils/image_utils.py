import numpy as np
from PIL import Image
import torchvision
def tensor2img(tensor, min_max=(-1, 1)):
    tensor = tensor.float().detach().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])
    img_np = tensor.numpy().transpose((1,2,0))
    img_np = (img_np * 255.0).round().astype(np.uint8)
    return img_np
def pad_to_ratio(img_np, ratio, pad_value, return_pad=False):
    ratio_w, ratio_h = ratio
    if len(img_np.shape) == 3:
        h, w, c = img_np.shape
        if c == 1:
            pad = [(0, 0), (0,0)]
            img_np = img_np[..., 0]
        else:
            pad = [(0,0), (0,0), (0,0)]
    else:
        h, w = img_np.shape
        pad = [(0,0), (0,0)]
    factor = ratio_h / ratio_w
    if h / w > factor:
        pad_w = int(h / factor - w)
        pad_head = pad_w // 2
        pad_tail = pad_w - pad_head
        pad[1] = (pad_head, pad_tail)
    else:
        pad_h = int(w * factor - h)
        pad_head = pad_h // 2
        pad_tail = pad_h - pad_head
        pad[0] = (pad_head, pad_tail)
    pad = tuple(pad)
    img_pad = np.pad(img_np, pad, mode='constant', constant_values=(pad_value, pad_value))
    if return_pad:
        return img_pad, pad
    return img_pad

def crop_to_ratio(img_pil, size, return_info=False, crop_lower=False):
    w, h = img_pil.size
    ratio_org = w/h
    ratio = size[0] / size[1]
    img_cropped = img_pil
    crop_x = 0
    crop_y = 0
    if ratio_org == ratio:
        pass
    elif ratio_org > ratio:
        target_w = int(h * ratio)
        border = (w - target_w) // 2
        if border == 0:
            pass
        else:
            img_cropped = img_pil.crop((border, 0, w - border, h))
            crop_x, crop_y = border, 0
    else:
        target_h = int(w / ratio)
        if not crop_lower:
            border = (h - target_h) // 2
            if border == 0:
                pass
            else:
                img_cropped = img_pil.crop((0,border, w, h - border))
                crop_x = 0
                crop_y = border
        else:
            border = h - target_h
            if border == 0:
                pass
            else:
                img_cropped = img_pil.crop((0, 0, w, h - border))
                crop_x = 0
                crop_y = 0

    if return_info:
        return img_cropped, crop_x, crop_y
    else:
        return img_cropped
def save_image_tensor(img_tensor, save_path, min_max=(-1, 1)):
    if img_tensor is None:
        return
    if img_tensor.dim() == 4:
        img_tensor = torchvision.utils.make_grid(img_tensor)
    img_np = tensor2img(img_tensor, min_max=min_max)
    if img_np.shape[-1] == 1:
        img_np = np.squeeze(img_np, -1)
    im = Image.fromarray(img_np)
    im.save(save_path)