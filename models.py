#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 14:31:33 2021

@author: fj
"""

from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as f

# SE_Block
class SEBlock(nn.Module):
    def __init__(self,channel=64,hidden_node=16):
        super().__init__()
        self.se_block = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Conv2d(channel,hidden_node,kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_node,channel,kernel_size=1),
                nn.Sigmoid(),
                )
        
    def forward(self,x,target=None):
        b,c,_,_ = x.shape
        channel_weights = self.se_block(x)
        if target == None:
            return channel_weights*x
        else:
            assert c == target.shape[1], "target\'s channels must equal to the input\'s"
            return channel_weights*target
        
# Cosine Similarity
class CosineSimilarity(nn.Module):
    def __init__(self,in_features=1024,out_features=31, momentum=0.9,is_center=False):
        super().__init__()
        self.weights = nn.Parameter(torch.Tensor(in_features,out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weights.data.normal_(0, 0.005)
        self.bias.data.normal_(10, 0.001)
        self.is_center = is_center
        
        if momentum: 
            self.register_buffer('old_weights', torch.Tensor(in_features,out_features))
            self.register_buffer('old_bias',  torch.Tensor(out_features))
            self.momentum = momentum
            self.old_weights.data.normal_(0, 0.005)
            self.old_bias.data.normal_(10, 0.001)
        else:
            self.momentum = None
    
    def forward(self,x:torch.Tensor):
        if self.momentum:
            self.weights.data = self.momentum*self.weights + (1-self.momentum)*self.old_weights
            self.bias.data = self.momentum*self.bias + (1-self.momentum)*self.old_bias
            self.old_weights = self.weights.clone().detach()
            self.old_bias = self.bias.clone().detach()
        
        
        if self.is_center:
            self.weights.data = self.weights - self.weights.mean(dim=0)
            x = x - x.mean(dim=0)
        
#        x = F.dropout(F.relu(x))
        return F.normalize(x).matmul(F.normalize(self.weights,dim=0))*self.bias # TODO: check
    
class Network(nn.Module):
    def __init__(self,num_classes=5, momentum=0.9, res_layer=50, use_CS=True, SE_block=True):
        super().__init__()
        self.use_se = SE_block
        if res_layer == 50:
            model = models.resnet50(pretrained=True)
            in_features = 2048
        else:
            model = models.resnet34(pretrained=True)
            in_features = 512
        self.feature_extractor = nn.Sequential(*list(model.children())[:-2])
        del model

        # SE_Block
        if SE_block:
            self.se_block = SEBlock(channel=in_features,hidden_node=int(in_features/4))
            self.se_block.apply(init_weights)
        if use_CS:
            self.linear = CosineSimilarity(in_features=in_features,out_features=num_classes,momentum=momentum)
        else:
            self.linear = nn.Linear(in_features,num_classes)
            self.linear.apply(init_weights)
    
    def forward(self,x):
        features = self.feature_extractor(x)
        if self.use_se:
            features = self.se_block(features)
        features = F.adaptive_avg_pool2d(features,(1,1))
        features = features.view(x.shape[0],-1)
        out = self.linear(features)
        return out


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)

class Network_wo_fc(nn.Module):
    def __init__(self,num_classes=5, momentum=0.9, res_layer=50, use_CS=True):
        super().__init__()
        
        if res_layer == 50:
            model = models.resnet50(pretrained=True)
            in_features = 2048
        else:
            model = models.resnet34(pretrained=True)
            in_features = 512
        self.feature_extractor = nn.Sequential(*list(model.children())[:-2])
        del model
        
        self.linear = nn.Sequential(
                nn.Linear(in_features*2,int(in_features/2)),
                nn.Dropout(p=0.2),
                nn.ReLU(inplace=False),
                nn.Linear(int(in_features/2),num_classes),
                                    )
        self.linear.apply(init_weights)

    def forward(self,x,mask):
#        features = self.feature_extractor(x)
#        mask = f.resize(mask,features.shape[-2:])
#        mask_feature = features.permute(1,0,2,3) * mask
#        mask_feature = mask_feature.permute(1,0,2,3)
#        features = torch.cat((features,mask_feature),dim=1)
#        features = F.adaptive_max_pool2d(features,(1,1)).reshape(features.shape[0],-1)
#        out = self.linear(features)
        img_f = self.feature_extractor(x)
        mask_f = self.feature_extractor(mask)
        features = torch.cat((img_f,mask_f),dim=1)
        features = F.adaptive_avg_pool2d(features,(1,1)).reshape(features.shape[0],-1)
#        img_f = F.adaptive_avg_pool2d(img_f,(1,1)).reshape(img_f.shape[0],-1)
#        mask_f = F.adaptive_avg_pool2d(mask_f,(1,1)).reshape(mask_f.shape[0],-1)
        
        out = self.linear(features.contiguous())
        return out
    
if __name__ == '__main__':
    img = torch.randn([2,3,224,224])
    mask = torch.randn([2,3,224,224])
    net = Network(use_CS=False)
#    mask = torch.randn([2,224,224])
    preds = net(img)
    