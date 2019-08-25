import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from PIL import Image
from utils import is_image_file
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='predicting and mosaicing masks')
parser.add_argument('--nepochs', type=int, default=200, help='saved model of which epochs')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()
print(opt)

def segmentation(img, model = None, patch_size = 256):
    w, h = img.size
    coord2patch = {}
    mosaic = Image.new(size = (w, h), mode = 'RGB')
    
    for x in range(0, w, patch_size):
        for y in range(0, h, patch_size):
            print("\tReading patch...", (x,y))
            coord2patch[str(x)+'_'+str(y)] = np.array(img.crop((x, y, x+patch_size, y+patch_size)))
    
    coords = [tuple(map(int, coord.split('_'))) for coord in coord2patch.keys()]
    patches = np.array(list(coord2patch.values()))
    
    seg_results = []
    
    if model is None:
        seg_results = patches
    else:
        print("using models")
        patches = transform(patches)
        patches = patches.to(device)
        seg_results = model(patches)
        seg_results = seg_results.detach().cpu()
        pass
    
    for idx, coord in enumerate(coords):
        print('\tWrting mask...', coord)
        mosaic.paste(Image.fromarray(seg_results[idx]), coord)
        
    return mosaic


image_pre_dir = "./"
image_filenames = [x for x in os.listdir(image_pre_dir) if is_image_file(x)]
image_save_dir = './'

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform = transforms.Compose(transform_list)


device = torch.device("cuda:0" if opt.cuda else "cpu")
model_path= "netG_model_epoch_{}.pth".format(opt.nepochs)
net_g = torch.load(model_path).to(device)
net_g = nn.DataParallel(net_g,device_ids=[0,1,2])


for idx, img_name in enumerate(image_filenames):
    img = Image.open(os.path.join(image_pre_dir, img_name))
    print('Processing %d/%d' % (idx+1, len(image_filenames)))
    seg = segmentation(img, model = net_g)
    seg.save(os.path.join(image_save_dir, "img_name.png"))