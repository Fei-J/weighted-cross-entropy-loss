#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 16:12:13 2021

@author: fj
"""
import os
import cv2
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd

class VOCDataset(Dataset):
    def __init__(self, images_filenames, images_directory, masks_directory, transform=None):
        self.images_filenames = images_filenames
        self.images_directory = images_directory
        self.masks_directory = masks_directory
        self.transform = transform

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, idx):
        image_filename = self.images_filenames['image'][idx]
        label = self.images_filenames['level'][idx]
        image = cv2.imread(os.path.join(self.images_directory, image_filename[:-3]+'png'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(
            os.path.join(self.masks_directory, image_filename)#, cv2.IMREAD_GRAYSCALE
        )
        mask = A.normalize(mask,mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            
        return image, mask, label

def data_loader(batch_size=32,img_size=512, crop_size=(448,448), num_workers=4,
                    path = '/home/fj/Desktop/PyTorch/dataset/Diabetic_Retinopathy/Optic_Disc/',
                    images_directory = '/home/fj/Desktop/PyTorch/dataset/Diabetic_Retinopathy/Optic_Disc/img/',
                    masks_directory = '/home/fj/Desktop/PyTorch/dataset/Diabetic_Retinopathy/Optic_Disc/seg/',
                    phase='train'):
    path = './list/'
    images_directory = '/home/fj/Desktop/PyTorch/dataset/Diabetic_Retinopathy/Messidor_m/'
    masks_directory = '/home/fj/Desktop/PyTorch/dataset/Diabetic_Retinopathy/Messidor_VesselSeg4/'
    if phase == 'train':
        train_transform = A.Compose(
                [
                A.Resize(img_size, img_size),
#                A.RandomCrop(crop_size,crop_size),
                A.RandomResizedCrop(crop_size[0],crop_size[1],scale=(0.7,2),interpolation=cv2.INTER_NEAREST),
#                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
                ])
        train_img_name = path + 'messidor_t0.2.csv'
        img_name = pd.read_csv(train_img_name)
        
        train_dataset = VOCDataset(img_name, images_directory, masks_directory, transform=train_transform)
        train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,
                             num_workers=num_workers, pin_memory=True,drop_last=True)
        return train_loader
    
    if phase == 'val':
        val_img_name = path + 'messidor_v0.2.csv'
        img_name = pd.read_csv(val_img_name)
        
        val_transform = A.Compose(
            [A.Resize(crop_size[0], crop_size[1],interpolation=cv2.INTER_NEAREST), 
             A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
             ToTensorV2(),
             ]
        )
        val_dataset = VOCDataset(img_name, images_directory, masks_directory, transform=val_transform)
        val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,
                             num_workers=num_workers, pin_memory=True,drop_last=False)
        return val_loader

if __name__ == '__main__':
    from utils import show_images
    loader = data_loader(batch_size=16,img_size=256,crop_size=(400,400),num_workers=0,phase='train')
    for image,GT,label in loader:
        show_images(image,num_cols=5,num_rows=1)
        show_images(GT.numpy(),num_cols=5,num_rows=1)
        break

            