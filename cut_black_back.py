#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 21:54:24 2021

@author: fj
"""
from PIL import Image
import numpy as np
import cv2
import os

def clahe(img,clipLimit=2.0, tileGridSize=(8,8), Color='RGB'):
    """
    img: RGB, unit8
    """
    clahe_ = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    
    # gray image
    if len(img.shape) == 2:
        cl = clahe_.apply(img)
        return cl
    
    # colorful image
    if Color == 'RGB':
        (R,G,B) = cv2.split(img)
        r = clahe_.apply(R)
        g = clahe_.apply(G)
        b = clahe_.apply(B)
        cl = cv2.merge([r,g,b])
        return cl
    
    if Color == 'LAB':
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        L = img_lab[:,:,0]
        cl = clahe_.apply(L)
        img_lab[:,:,0] = cl
        return cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
    
    if Color == 'HSV':
        img_hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        V = img_hsv[:,:,2]
        cl = clahe_.apply(V)
        img_hsv[:,:,2] = cl
        return cv2.cvtColor(img_hsv,cv2.COLOR_HSV2RGB)

def find_max_gap(vector:list):
    vector = np.array(vector)
    gap = vector[1:]-vector[:-1]
    index = np.argmax(gap)
    return [vector[index],vector[index+1]]

def find_point(vector):
    points = []
    for i in range(len(vector)-1):
        if vector[i] != vector[i+1]:
            points.append(i)
    return points
        
file_path = '/home/fj/Desktop/PyTorch/dataset/Diabetic_Retinopathy/Messidor/'
save_path = '/home/fj/Desktop/PyTorch/dataset/Diabetic_Retinopathy/Messidor_/'

for file_name in os.listdir(file_path):
    
    try:
        path = file_path+file_name
        img = Image.open(path)
        img_L = np.array(img.convert('L'))
        mean = np.sum(img_L)/np.prod(img_L.shape) * 10
        img_arr = np.sum(np.array(img),axis=2)
        img_arr[img_arr<min(mean,30)] = 0
        
        col_sum = np.sum(img_arr,axis=0) < 1000
        row_sum = np.sum(img_arr,axis=1) < 1000
        
        x = find_point(col_sum)
        y = find_point(row_sum)
        
        x.insert(0,0)
        x.append(len(col_sum))
        y.insert(0,0)
        y.append(len(row_sum))
        
        x = find_max_gap(x)
        y = find_max_gap(y)
        
        # crop image to 1000*1000
        img = img.crop((x[0],y[0],x[-1],y[-1]))
        img = img.resize((1000,1000))
        
        img = np.array(img)
        
        # clahe
#        img = clahe(img,Color='LAB')
        
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path+file_name,img)
    except:
        print(path)
        continue


import os
import cv2
path = '/tmp/Messidor_/'
for name in os.listdir(path):
    img = cv2.imread(path+name)
    cv2.imwrite(path+name[:-3]+'png')
