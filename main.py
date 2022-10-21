#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 09:43:21 2021

@author: fj
"""
import os
import argparse
import numpy as np
import math
import torch
import torch.optim as optim
import torch.nn as nn
from utils import data_loader,LabelSmoothingCrossEntropy_weighted
from utils import test,auc,test_binary
from models import Network
from torch.cuda.amp import autocast,GradScaler
from tqdm import tqdm
from tensorboardX import SummaryWriter
from datetime import datetime
#%% Command setting
# 
parser = argparse.ArgumentParser(description='Diabetic Retinopathy')
parser.add_argument('-batch_size', '-b', type=int, help='batch size', default=32)
parser.add_argument('-cuda', '-g', type=int, help='cuda id', default=0)
parser.add_argument('-Epoch', '-e', type=int, default=200)

# learning rate
parser.add_argument('-lambda_lr', '-llr',type=str, default='cos_lr')
parser.add_argument('-learning_rate', '-lr', type=float, help='learning rate', default=1e-4)
parser.add_argument('-warm_up_epochs', '-w', type=int, help='warm up epoch for Cosine Schedule', default=5)
parser.add_argument('-weight_decay', '-wd', type=float, default=4e-5,
                    help='weight decay for Adam')
# dataset
parser.add_argument('-data_path', '-path', type=str, 
                    default='/home/fj/Desktop/PyTorch/dataset/Diabetic_Retinopathy/Messidor_m/',
                    help='training and testing datasets\'s path')
parser.add_argument('-train_file', '-train', type=str, default='./list/messidor_t0.2.csv')
parser.add_argument('-test_file', '-test', type=str, default='./list/messidor_v0.2.csv')
parser.add_argument('-img_size', '-is', type=int, default=512)
parser.add_argument('-crop_size', '-cs', type=int, default=448)

args = parser.parse_args()

#%% record args
save_file_name = os.listdir('./output')

with open('./output/'+str(len(save_file_name))+'.txt', 'w') as f:
    f.write(str(args))
print(str(args))

#%% randomm seed
seed = 2021
print(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#%% setting
batch_size = args.batch_size
n_epoch = args.Epoch
lr = args.learning_rate
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
weight_decay = args.weight_decay
data_path= args.data_path
lambda_lr = args.lambda_lr

# tensorboard log-dir
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
tb_writer = SummaryWriter(log_dir='./runs/'+TIMESTAMP)
tb_writer.add_text(tag="super parameters",text_string=str(args),global_step=0)

#%% model and optimizer
#model = FeaturePyramidNetwork()
model = Network(num_classes=5,use_CS=False,SE_block=True)
model.to(DEVICE)
#model.load_state_dict(torch.load('./checkpoint_50.pth'),strict=False)

# write model structure into tensorboard
init_img = torch.zeros((1,3,args.crop_size,args.crop_size),device=DEVICE)
tb_writer.add_graph(model,init_img)


#optimizer = optim.SGD(model.parameters(),lr=lr,weight_decay=weight_decay)
optimizer = optim.Adam([{'params':model.feature_extractor.parameters(),'lr':lr},
                         {'params':model.se_block.parameters(),'lr':lr},
                         {'params':model.linear.parameters(), 'lr':lr}
                         ], betas=(0.9,0.999),weight_decay=weight_decay)
if lambda_lr == 'step_lr':
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=50)

else:
# Learning rate update schedulers
    warm_up_epochs = args.warm_up_epochs
    warm_up_with_cosine_lr = lambda epoch:  (epoch+1) / warm_up_epochs if epoch < warm_up_epochs \
    else  0.5 * ( math.cos((epoch - warm_up_epochs) /(n_epoch - warm_up_epochs) * math.pi) + 1)
    
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,warm_up_with_cosine_lr)

criterion = LabelSmoothingCrossEntropy_weighted(num_classes=5,weight='linear',smoothing=0.2).to(DEVICE)
#criterion = nn.BCEWithLogitsLoss().to(DEVICE)
#criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(DEVICE) 
#%% training and testing process
train_loader = data_loader(img_size=args.img_size,crop_size=args.crop_size, csv_name=args.train_file, 
                           path=data_path, batch_size=batch_size,num_workers=8)

val_loader = data_loader(phase='validation', img_size=args.img_size,crop_size=args.crop_size,
                         csv_name=args.test_file,shuffle=False,
                         path=data_path, batch_size=batch_size, num_workers=8,with_path=True)
best_acc = 0
current_acc = 0

scaler = GradScaler()

for epoch in range(n_epoch):
    length = len(train_loader)
    total_loss = 0.
    with tqdm(total=length,postfix=dict,mininterval=0.3) as pbar:
        for i, data in enumerate(train_loader):
            img,label = data[0].to(DEVICE),data[1].long().to(DEVICE)
            # mixup
#            alpha=0.2
#            lam = np.random.beta(alpha,alpha)
#            index = torch.randperm(img.size(0)).cuda()
#            inputs = lam*img + (1-lam)*img[index,:]
#            targets_a, targets_b = label, label[index]

            with autocast():
                preds = model(img)
                preds = preds.squeeze()
#                loss = lam * criterion(preds, targets_a) + (1 - lam) * criterion(preds, targets_b)
                loss = criterion(preds,label)
            total_loss = total_loss + loss.detach().cpu().numpy()
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # print log
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            pbar.set_description(f'epoch:{epoch+1}/{n_epoch}, iter:{i + 1}/{length}')
            pbar.set_postfix(**{'avg_loss': total_loss/(i+1),
                                'lr'        : current_lr})
            pbar.update(1)
#            break
        lr_scheduler.step()
        
        # add avg_loss and lr into tensorboard
        tb_writer.add_scalar("avg_loss", total_loss/(i+1), epoch)
        tb_writer.add_scalar("lr", current_lr, epoch)
    
    #TODO add figure into tensorboard
#    fig = plot_class_preds(net=model,
#                           images_dir='./demo_img',
#                           transformer=None,
#                           num_plot=5,
#                           device=DEVICE)
#    if fig is not None:
#        tb_writer.add_figure("preds vs real", 
#                             figure=fig,
#                             global_step=epoch)
    # add weights into tensorboard
    tb_writer.add_histogram(tag="conv1",
                            values=model.linear.weight,
                            global_step=epoch)
    #validation
    if epoch % 10 == 0 or epoch == n_epoch - 1:

        t_loss,acc,acc_pre,kappa_score,labels,all_preds,all_output = test(model, val_loader, DEVICE)
        fpr, tpr, auc_v = auc(labels,all_preds)
        print("t_loss: %f, acc: %f\n" % (t_loss, acc), acc_pre, " ",kappa_score,' ',auc_v)
        
        # add acc, kappa and auc_v into tensorboard 
        tb_writer.add_scalar("acc:", acc, epoch)
        tb_writer.add_scalars(main_tag="acc_pre", tag_scalar_dict=
                              dict(zip([str(i) for i in acc_pre.keys()],acc_pre.values)), global_step=epoch)
        tb_writer.add_scalar("kappa:",kappa_score, epoch)
        tb_writer.add_scalar("auc_v:",auc_v, epoch)
        
        current_acc = acc
        if current_acc >= best_acc:
            best_acc = current_acc
            best_pre = acc_pre
            torch.save((model.state_dict(),optimizer.state_dict()),'./checkpoint/checkpoint_'+str(len(save_file_name))+'.pth')
        
        with open('./output/'+str(len(save_file_name))+'.txt', 'a') as f:
            f.write('\nepoch: {},t_loss: {:.6f}, acc: {:.4f}\n'.format(epoch, t_loss, acc))
            f.write(acc_pre.to_string())
            f.write('\nkappa_score:'+str(kappa_score))
os.rename('./output/'+str(len(save_file_name))+'.txt','./output/'+str(len(save_file_name))+'_{:.4f}.txt'.format(best_acc))
            
