from __future__ import print_function
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from utils import is_image_file, load_img, save_img

# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
#which dataset to use
#parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--nepochs', type=int, default=200, help='saved model of which epochs')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if opt.cuda else "cpu")

#where the model is stored
#model_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, opt.nepochs)
model_path = "netG_model_epoch_{}.pth".format(opt.nepochs)

net_g = torch.load(model_path).to(device)
net_g = nn.DataParallel(net_g,device_ids=[0,1,2])

#image_dir = "dataset/{}/test/images/".format(opt.dataset)
image_pre_dir = "/media/hdd1/MUTTER_WSI/patches_sn/"

image_filenames = [x for x in os.listdir(image_pre_dir) if is_image_file(x)]

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)

for image_name in image_filenames:
    img = load_img(image_pre_dir + image_name)
    img = transform(img)
    input = img.unsqueeze(0).to(device)
    out = net_g(input)
    out_img = out.detach().squeeze(0).cpu()

#     if not os.path.exists(os.path.join("result", opt.dataset)):
#         os.makedirs(os.path.join("result", opt.dataset))
#     save_img(out_img, "result/{}/{}".format(opt.dataset, image_name))
    if not os.path.exists(os.path.join(image_pre_dir, "result")):
        os.makedirs(os.path.join(image_pre_dir, "result"))
    save_img(out_img, image_pre_dir+"result/{}".format(opt.dataset, image_name))