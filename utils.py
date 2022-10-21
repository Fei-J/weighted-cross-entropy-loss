#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 14:18:59 2021

@author: fj
"""
import pandas as pd
import numpy as np
import torch.nn as nn
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision import datasets
from sklearn.metrics import roc_auc_score,cohen_kappa_score,accuracy_score,precision_score,recall_score

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# pre-process:subtract local mean color by file:///home/fj/Downloads/competitionreport.pdf
def subtract_local_mean(img:Image,alpha=4):
    img = np.array(img)
    img = cv2.addWeighted(img,alpha,cv2.medianBlur(img,31),-alpha,128)
    return Image.fromarray(img)

# label smoothing
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        print(logprobs.shape)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1) # cross_entropy loss without mean
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()

class LabelSmoothingCrossEntropy_weighted(nn.Module):
    def __init__(self,num_classes:int=4, weight:str=None, smoothing=0.1):
        super(LabelSmoothingCrossEntropy_weighted, self).__init__()
        self.num_classes = num_classes
        self.weight=weight
        self.smoothing = smoothing
        
    def forward(self, x, target):
#        confidence = 1. - self.smoothing
        
        if self.weight == None:
            return F.cross_entropy(x,target,label_smoothing=self.smoothing)
        
        if self.weight == 'quadratic':
            weight = self._create_weight(target,self.num_classes).to(x.device)
            
        if self.weight == 'linear':
            weight = self._create_weight(target,self.num_classes).to(x.device)
        
        logprobs = x.log_softmax(dim=-1)
        loss = -logprobs * weight
        return loss.mean()

    
    def _create_weight(self,target,num_classes=-1):
        if num_classes == -1:
            num_classes = torch.max(target) + 1
            
        # prepare for weights
        if self.weight == 'quadratic':
            pre_weight = [(i+1)**2 for i in range(num_classes)]
        if self.weight == 'linear':
            pre_weight = [(i+1) for i in range(num_classes)]
        pre_weight.extend(pre_weight[::-1][1:])
        
        # one hot with smoothing
        one_hot = F.one_hot(target,num_classes)
        one_hot = (1. - self.smoothing) * one_hot + self.smoothing * torch.ones_like(one_hot)/num_classes
        
        weight = [pre_weight[-(num_classes):] 
                    if label == 0 else pre_weight[-(num_classes+label):-label]
                    for label in target]
        weight = torch.Tensor(weight).to(target.device)
        
        # weight with normalized
        weight = one_hot*weight/torch.sum(one_hot*weight,dim=1,keepdim=True)
        
        return weight
        

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, weight = None):
        """if smoothing == 0, it's one-hot method
           if 0 < smoothing < 1, it's smooth method
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=self.dim)

        if self.weight is not None:
            pred = pred * self.weight.unsqueeze(0)   

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
    
#%% dataloder with path
class MyDataset(Dataset):
    def __init__(self,list_name='trainLabels.csv',path='/home/data/kaggle_diabetic_retinopathy/train/',
                 transform=None,with_path=False,read_all=False):
        super().__init__()
        self.img_name = pd.read_csv(list_name)
        self.transform = transform
        self.path = path
        self.with_path =with_path
        self.read_all = read_all
        if self.read_all:
            self.total_img = []
            self.total_label = self.img_name['level']
            for i in range(len(self.img_name)):
                form = '.png' if 'aptos' in self.path else '.jpeg'
                img_path = self.path + str(self.img_name['image'][i])+ form
                img = Image.open(img_path)
#                print(img.size)
                self.total_img.append(img)
#            print(len(self.total_img),type(self.total_img))
    
    def __getitem__(self,index):
        form = '.png' if 'aptos' in self.path else '.jpeg'
        
        if self.read_all:
            img = self.total_img[index]
            label = self.total_label[index]
            path = self.path + str(self.img_name['image'][index])+ form
        else:
#            path = self.path + str(self.img_name['image'][index])+ form
            path = self.path + str(self.img_name['image'][index])#[:-3]+'png'
            label = self.img_name['level'][index]
            img = Image.open(path)
            img = img.convert('RGB')
#            img = subtract_local_mean(img) # subtract_local_mean
            
        if self.transform:
            img = self.transform(img)
            
        if self.with_path:
            return img,label,path
        else:
            return img, label
    
    def __len__(self):
        return len(self.img_name)

def data_loader(batch_size=32,img_size=512, crop_size=448, num_workers=4,csv_name='train.csv', 
                path='/home/data/kaggle_diabetic_retinopathy/train/',phase='train',
                with_path=False, shuffle=True,read_all=False):
    if phase == 'train':
        transform = transforms.Compose([
                transforms.Resize((img_size,img_size)),
                transforms.RandomHorizontalFlip(),
#                transforms.RandomVerticalFlip(),
                transforms.RandomResizedCrop((crop_size,crop_size)),
#                transforms.RandomApply(nn.ModuleList([
#                                        transforms.ColorJitter(0.3,0.3,0.3),
#                                        ]),p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
#                transforms.Normalize([0.3202907 , 0.22441998, 0.16101242],
#                                     [0.30286581, 0.21897724, 0.17457085])
                ])
        dataset = MyDataset(list_name=csv_name, path=path, transform=transform,
                            with_path=with_path,read_all=read_all)
    else:
        transform = transforms.Compose([
                transforms.Resize((img_size,img_size)),
                transforms.CenterCrop((crop_size,crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
#                transforms.Normalize([0.3202907 , 0.22441998, 0.16101242],
#                                     [0.30286581, 0.21897724, 0.17457085])
                ])
        dataset = MyDataset(list_name=csv_name,path=path,transform=transform,
                            with_path=with_path,read_all=read_all)
        
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,
                             num_workers=num_workers, pin_memory=True,drop_last=False)
    return dataloader


#%% focal_loss
import torch
from torch import nn
from torch.nn import functional as F

class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        """
        focal_loss损失函数, $-\alpha (1-\hat{y})^{\gamma} * CrossEntropyLoss(\hat{y}, y)$
        alpha: 类别权重. 当α是列表时, 为各类别权重, 当α为常数时, 类别权重为[α, 1-α, 1-α, ....]
        gamma: 难易样本调节参数.
        num_classes: 类别数量
        size_average: 损失计算方式, 默认取均值
        """
        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes   # α可以以list方式输入, 用于对不同类别精细地赋予权重
            print(" --- Focal_loss alpha = {}, 将对每一类权重进行精细化赋值 --- ".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            print(" --- Focal_loss alpha = {} --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为[α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        preds: 预测类别. size:[B, C] or [B, S, C] B 批次, S长度, C类别数
        labels: 实际类别. size:[B] or [B, S] B批次, S长度
        """
        # assert preds.dim() == 2 and labels.dim()==1
        labels = labels.view(-1, 1) # [B * S, 1]
        preds = preds.view(-1, preds.size(-1)) # [B * S, C]
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) # 先softmax, 然后取log
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        print(labels)
        print(preds_softmax)
        preds_softmax = preds_softmax.gather(1, labels)   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels)
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
      

def test_binary(model:torch.nn.Module, dataloader:torch.utils.data.DataLoader,DEVICE):
    criterion = torch.nn.BCEWithLogitsLoss().to(DEVICE)
    with torch.no_grad():
        total_loss,correct, acc = 0,0,0
        all_label,all_preds = [], []
        model.eval()
#        length = len(dataloader.dataset)
        
        for img,label in dataloader:
            img,label = img.to(DEVICE),label.long().to(DEVICE).float()
            output = model(img)
            output = output.view(output.shape[0])
            loss = criterion(output, label)
            pred = torch.sigmoid(output)
            pred = torch.where(pred>=0.5,1.,0.)
            all_label.extend(list(label.cpu().numpy()))
            all_preds.extend(list(pred.detach().cpu().numpy()))
#            correct += torch.sum(pred == label)
            total_loss += float(loss.item()) * img.shape[0]
                
#        acc = correct.double() / length
        acc = accuracy_score(all_label,all_preds)
        acc_pre = acc_pre_class(all_label,all_preds)
        kappa_score = cohen_kappa_score(all_label,all_preds,weights='quadratic')
    return total_loss, acc, acc_pre, kappa_score,all_label,all_preds
    
#%% accuracy for each classification
def acc_pre_class(real_label,preds_label):
    real_statistics = pd.value_counts(real_label,sort=False)
    N_class = len(real_statistics)
    acc_pre = dict(zip(range(N_class),[0]*N_class))
    for i in range(len(real_label)):
        acc_pre[real_label[i]] += real_label[i]==preds_label[i]
        
    return pd.Series(data=acc_pre).div(real_statistics)

def test(model:torch.nn.Module, dataloader:torch.utils.data.DataLoader,DEVICE):
#    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    with torch.no_grad():
        total_loss,correct, acc = 0,0,0
        all_label,all_preds,all_output = [], [], []
        model.eval()
#        length = len(dataloader.dataset)
        
#        for img,mask,label in dataloader:
#            img,mask,label = img.to(DEVICE),mask.to(DEVICE),label.long().to(DEVICE)
#            mask = mask.permute(0,3,1,2)
        for img,label,_ in dataloader:
            img,label = img.to(DEVICE),label.long().to(DEVICE)
            output = model(img)
            
            loss = criterion(output, label)
            pred = torch.max(output,1)[1]
            all_label.extend(list(label.cpu().numpy()))
            all_preds.extend(list(pred.detach().cpu().numpy()))
            all_output.extend(list(output.detach().cpu().numpy()))
#            correct += torch.sum(pred == label)
            total_loss += float(loss.item()) * img.shape[0]
        
#        auc = roc_auc_score(all_label,all_preds,multi_class='ovr')
#        acc = correct.double() / length
        acc = accuracy_score(all_label,all_preds)
        acc_pre = acc_pre_class(all_label,all_preds)
#        kappa_score = quadratic_weighted_kappa(all_label,all_preds)
        kappa_score = cohen_kappa_score(all_label,all_preds,weights='quadratic')
    return total_loss, acc, acc_pre, kappa_score,all_label,all_preds,all_output

# auc
def auc(y_true,scores,pos_label=1):
    from sklearn import metrics    
    fpr, tpr, thresholds = metrics.roc_curve(y_true, scores, pos_label=pos_label)
    return fpr, tpr, metrics.auc(fpr, tpr)


#%% quadratic_weighted_kappa
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None: 
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat
 
def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings

def quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):

    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / denominator

# calculate dataset's mean and std, mean = sum(x_i)/N, std^2 = sum(x^2)/N-mean^2
def mean_std(path='/home/data/kaggle_diabetic_retinopathy/train/'):
    import os
    import numpy as np
    from PIL import Image
    
    img_names = os.listdir(path)
    img_sum,square_sum = np.array([0.,0.,0.]),np.array([0.,0.,0.])
    N_pixels = 0
    for name in img_names:
        img = Image.open(path+name)
        N_pixels += np.product(img.size)
        img_np = np.array(img)/255
        img_sum += np.sum(img_np, axis=(0,1))
        square_sum += np.sum(img_np*img_np,axis=(0,1))
    mean = img_sum/N_pixels
    std = np.sqrt(square_sum/N_pixels - mean*mean)
    return mean, std
    
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images.

    Defined in :numref:`sec_fashion_mnist`"""
    import matplotlib.pyplot as plt
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
        if titles:
            ax.set_title(titles[i])
    return axes    

        
if __name__ == '__main__':
#    from models import Network
#    
#    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#    model = Network()
#    model.load_state_dict(torch.load('./checkpoint/checkpoint_7.pth')[0])
#    model = model.to(DEVICE)
    
#    def data_loader_val(path='/home/data/ILSVRC2012_img_train', batch_size=32,
#                              img_size=512, crop_size=448,num_worker=8):
#        # data loader
#        # Image transformations
#        train_transform = transforms.Compose([
#                transforms.Resize((img_size,img_size)),
#                transforms.CenterCrop((crop_size,crop_size)),
#                transforms.ToTensor(),
#                ])
#                
#        
#        # Training data loader
#        train_dataset = datasets.ImageFolder(root=path, transform=train_transform)
#        dataloader_train = DataLoader(train_dataset,batch_size=batch_size, shuffle=True,
#                                      num_workers=num_worker, pin_memory=True)
#        return dataloader_train
    path = '/home/data/aptos2019_clahe/'
#    path = '/home/fj/Desktop/PyTorch/dataset/Diabetic_Retinopathy/zhang/'
    csv_name = './list/aptos2019_train0.csv'
#    csv_name = 'validation_clahe.csv'
#    test_loader =  data_loader(path=path,csv_name=csv_name,phase='validation',read_all=True)
#    for img, label in test_loader:
#        print(img.shape)
#        break

    
    from torch.autograd import Variable
    
    crit = LabelSmoothingCrossEntropy_weighted(num_classes=5,weight='quadratic',smoothing=0.3)
    predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                                 [0, 0.9, 0.2, 0.2, 1], 
                                 [1, 0.2, 0.7, 0.9, 1]])
    label = Variable(torch.LongTensor([2, 1, 0]))
    v = crit(Variable(predict),label)
    print(v)
    
