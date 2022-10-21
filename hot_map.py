#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 23:12:32 2021

@author: fj
"""

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import preprocess_image
import cv2 
import torch
from models import Network
from utils import data_loader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import resnet50
import pandas as pd
import os

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    if len(img.shape) > 3:
        b,h,w,c = img.shape
        heatmap = []
        for i in range(b):
            heatmap_i = cv2.applyColorMap(np.uint8(255 * mask[i]), colormap)
            if use_rgb:
                heatmap_i = cv2.cvtColor(heatmap_i, cv2.COLOR_BGR2RGB)
            heatmap.append(heatmap_i)
        
        heatmap = np.stack(heatmap) / 255
        
    else:
        heatmap = cv2.applyColorMap(np.uint8(255 * mask[0]), colormap)
        if use_rgb:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255
        img = np.float32(img) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images.

    Defined in :numref:`sec_fashion_mnist`"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # Tensor Image
            ax.imshow(img.numpy())
        else:
            # PIL Image
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles is not None:
            ax.set_title(titles[i])
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    return axes

#model = resnet50(pretrained=True)

def hot_map(model, target_layers, use_cuda,loader=None, img_path=None):
        
    if loader != None and img_path == None:
        input_tensor,label,path = iter(loader).next()
        #
        img_raw = []
        for i,img_path in enumerate(path):
            img_ = Image.open(img_path)
            transform = transforms.Compose([transforms.Resize((800,800)),transforms.CenterCrop((720,720))])
            img_raw.append(np.array(transform(img_),dtype=np.float32)/255)
        img_raw = np.stack(img_raw)
        
    if loader == None and img_path != None:
        img_raw = cv2.imread(img_path)
        img_raw = cv2.cvtColor(img_raw,cv2.COLOR_BGR2RGB)
        input_tensor =  preprocess_image(img_raw)
        
    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=use_cuda)
    target_category = None
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
    visualization = show_cam_on_image(img_raw, grayscale_cam, use_rgb=True)
    return visualization

# Note: input_tensor can be a batch tensor with several images!
#path = '/home/fj/Desktop/PyTorch/dataset/Diabetic_Retinopathy/kaggle_1000/10_left.jpeg'

use_cuda = True if torch.cuda.is_available() else False
model = Network(use_CS=False)
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load('./checkpoint/checkpoint_351.pth')[0])
#target_layers = [model.layer4[-1]]
target_layers = [model.module.feature_extractor[7][-1]]
save_error_path = '/home/fj/Desktop/PyTorch/dataset/Diabetic_Retinopathy/messidor_hotmap/'
#df = pd.read_csv('./error_kwod.csv')
#for i,path in enumerate(df['path']):
#    img = cv2.imread(path)
#    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#    visualization = hot_map(img_path=path,model=model,target_layers=target_layers,use_cuda=use_cuda)
#    plt.title("label:{} pred:{}".format(df['label'][i],df['pred'][i]))
#    plt.imshow(np.hstack((img,visualization)))
#    plt.savefig(save_error_path+path.split('/')[-1], bbox_inches='tight',dpi=300)
#    break
#    break

data_path = '/home/fj/Desktop/PyTorch/dataset/Diabetic_Retinopathy/Messidor_/'
#val_file = 'validation_clahe.csv'
#val_loader = data_loader(phase='validation', img_size=256,crop_size=224,
#                         csv_name=val_file,  path=data_path, batch_size=9, 
#                         num_workers=16,shuffle=True,with_path=True)
for i,img_name in enumerate(os.listdir(data_path)):
    img_path = data_path+img_name
    img = plt.imread(img_path)
#    visualization = hot_map(img_path=img_path,model=model,target_layers=target_layers,use_cuda=use_cuda)
#    img_plot = show_images(visualization,3,3)
    plt.axis('off')
    plt.imshow(img)
    plt.savefig(save_error_path+img_name[:-3]+'.png', bbox_inches='tight',dpi=300)
    if i > 20:
        break
    i += 1
