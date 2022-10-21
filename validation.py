#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 19:08:12 2021

@author: fj
"""
import torch
from models import Network
from utils import data_loader,acc_pre_class, quadratic_weighted_kappa
from sklearn import metrics
from scipy.special import softmax

def test(model:torch.nn.Module, dataloader:torch.utils.data.DataLoader,DEVICE):
#    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    with torch.no_grad():
        total_loss,correct, acc = 0,0,0
        all_output,all_label,all_preds,all_path =[], [], [], []
        model.eval()
        length = len(dataloader.dataset)
        
        for img,label,path in dataloader:
            img,label = img.to(DEVICE),label.long().to(DEVICE)
            output = model(img)
#            output = output.view(output.shape[0])
            loss = criterion(output, label)
#            pred = output.ge(0.5).float()
            pred = torch.max(output,1)[1]
            all_output.extend(list(output.detach().cpu().numpy()))
            all_label.extend(list(label.cpu().numpy()))
            all_preds.extend(list(pred.detach().cpu().numpy()))
            all_path.extend(path)
            correct += torch.sum(pred == label)
            total_loss += float(loss.item()) * img.shape[0]
                
        acc = correct.double() / length
        acc_pre = acc_pre_class(all_label,all_preds)
        kappa_score = quadratic_weighted_kappa(all_label,all_preds)
#        print(all_label)
#        print(all_preds)
    return total_loss, acc, acc_pre, kappa_score,all_output,all_label,all_preds,all_path
def metric_report(result):
    outputs = np.array(result[4])
    scores = np.sum(softmax(outputs,axis=1)[:,2:],axis=1)
    labels = [0 if label<=1 else 1 for label in result[5]]
    preds = [0 if label<=1 else 1 for label in result[6]]
    acc = metrics.accuracy_score(labels,preds)
    pre = metrics.precision_score(labels,preds)
    rec = metrics.recall_score(labels,preds)
    f1 = metrics.f1_score(labels,preds)
    auc = metrics.roc_auc_score(labels,scores)
    return auc, acc, pre, rec, f1

ckp = './checkpoint/checkpoint_351.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Network(num_classes=5,use_CS=False)
net = torch.nn.DataParallel(net)
net.load_state_dict(torch.load(ckp)[0])
net.to(DEVICE)

#data_path = '/home/fj/Desktop/PyTorch/dataset/Diabetic_Retinopathy/kaggle_1000/'
data_path = '/home/fj/Desktop/PyTorch/dataset/Diabetic_Retinopathy/Messidor_/'
val_file = './list/messidor_v0.2.csv'
val_loader = data_loader(phase='validation', img_size=800,crop_size=720,
                         csv_name=val_file,  path=data_path, batch_size=32, 
                         num_workers=8, with_path=True)


result = test(net, val_loader,DEVICE)
print(result[:4])
import numpy as np
import pandas as pd
label,pred,dir_list = np.array(result[5:])
index = ~(label == pred)
error = pd.DataFrame(data={'path':dir_list[index],'label':label[index],'pred':pred[index]})
error.to_csv('./error_kwod.csv',index=None)
error_matrix = metrics.confusion_matrix(label.astype(np.float),
                                pred.astype(np.float),labels=[0,1,2,3])
report = metric_report(result)
