import numpy as np
import torch 
import cv2
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    img = img.resize((256, 256), Image.BICUBIC)
    return img


def save_img(image_tensor, filename):
    image_tensor = torch.where(image_tensor>0, torch.full_like(image_tensor, 1), torch.full_like(image_tensor, 0))
    image_tensor = image_tensor.int()
    image_numpy = image_tensor.numpy()
    np.where(image_numpy>0, 1, 0)
#     print(np.unique(image_numpy))
    image_numpy  = np.transpose(image_numpy, (1, 2, 0))
    image_numpy *= 255
#     print(image_numpy.shape)
    
#    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 
#    image_numpy = image_numpy.clip(0, 1)
#     image_numpy = image_numpy.astype(np.uint8)
#     image_pil = Image.fromarray(image_numpy)
#     image_pil.save(filename)
    cv2.imwrite(filename, image_numpy)
    print("Image saved as {}".format(filename))
