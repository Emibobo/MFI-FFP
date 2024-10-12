'''
Author: Alberto
Date: 2024-09-10 15:57:27
LastEditors: Alberto
LastEditTime: 2024-10-12 16:18:19
Description: 
'''
import os
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'#set your CUDA_VISIBLE_DEVICES here
import math
import argparse
import pandas as pd 
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler 
from torch.utils.tensorboard import SummaryWriter
from Mydataset import get_data_loader,val_data_loader
from model import classifier
from utils import train_one_epoch,test_one_epoch,Customized_Loss,valid_one_epoch
import torch.nn as nn
import numpy as np
import random


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
def model_args():
    parser=argparse.ArgumentParser(description='prediction of lymph metastasis based on rgb,icg and color images')
    parser.add_argument('-epochs',type=int,default=100,help='number of epochs to train')
    parser.add_argument('-batch_size',type=int,default=32,help='number of batch size')
    parser.add_argument('-train_lir',type=str,default='',help='path to train lir')#set your path to train set  here
    parser.add_argument('-test_lir',type=str,default='',help='path to test lir')#set your path to test set here
    parser.add_argument('-val_lir',type=str,default='',help='path to test lir')#set your path to validation set here
    parser.add_argument('-data_path_rgb',type=str,default='',help='path to rgb data')#set your path to white light imageing data here
    parser.add_argument('-data_path_icg',type=str,default='',help='path to icg data')#set your path to fluorescence imaging data here
    parser.add_argument('-data_path_color',type=str,default='',help='path to color data')#set your path to pseudo-color imaging data here
    parser.add_argument('-val_rgb',type=str,default='',help='path to val_rgb data')#set your path to white light imageing data in validation set here
    parser.add_argument('-val_icg',type=str,default='',help='path to val_icg data')#set your path to fluorescence imaging data in validation here
    parser.add_argument('-val_color',type=str,default='',help='path to val_color data')#set your path to pseudo-color imaging data in validation  here
    parser.add_argument('-num_workers',type=int,default=8,help='number of workers for dataloader')
    parser.add_argument('-lr',type=float,default=0.001,help='learning rate')
    parser.add_argument('-wd',type=float,default=0.003,help='weight decay')
    parser.add_argument('-num_classes',type=int,default=2,help='number of classes')
    parser.add_argument('-out_dirc',type=str,default='',help='path to tensorboard')
    parser.add_argument('-seed',type=int,default=1,help='random seed for reproducibility')
    parser.add_argument('-loss_weight_alpha',type=float,default=1.0,help='hyperparameter analysis for loss')
    parser.add_argument('-loss_weight_beta',type=float,default=1.0,help='hyperparameter analysis for loss')
    args=parser.parse_args()
    return args
def main(args):
    
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    else:
        torch.manual_seed(args.seed)  

    tb_writer = SummaryWriter(log_dir=args.out_dirc)
    train_df=pd.read_csv(args.train_lir)
    test_df=pd.read_csv(args.test_lir)
    val_df=pd.read_csv(args.val_lir)
    train_loader=get_data_loader(args,train_df['img'],train_df['label'])
    test_loader=get_data_loader(args,test_df['img'],test_df['label'])
    val_loader=val_data_loader(args,val_df['img'],val_df['label'])
    model=classifier(num_classes=args.num_classes).to(device)
    optimizer=optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.wd)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, last_epoch=-1)    
    criterion = Customized_Loss(args)

    all_result = pd.DataFrame()
    best_test_auc, best_test_epoch = 0, 0
    best_val_auc, best_val_epoch = 0, 0

    #train, test, and validate the model for 100 epochs
    for epoch in range(args.epochs):
        train_result,train_loss,train_acc,train_auc,train_sen, train_spe, train_prec, train_f1=train_one_epoch(model=model,criterion=criterion,optimizer=optimizer,data_loader=train_loader, writer=tb_writer, epoch=epoch)
        scheduler.step()
         
        test_result,test_loss,test_acc,test_auc, test_sen, test_spe, test_prec, test_f1= test_one_epoch(model=model,criterion=criterion,data_loader=test_loader,writer=tb_writer,epoch=epoch) 
        
        val_result,val_loss,val_acc,val_auc,val_sen, val_spe, val_prec, val_f1 = valid_one_epoch(model=model,criterion=criterion,data_loader=val_loader,writer=tb_writer,epoch=epoch)
       
       #save the result and model weights pre epoch
        result = pd.concat([train_result,test_result,val_result])
        result.to_csv(args.out_dirc+'epoch_result/'+'epoch{}_result.csv'.format(epoch+1),index=0)
        torch.save(model.state_dict(), '%s/%s/epoch_%03d.pth' % (args.out_dirc,'weight',epoch+1))
    #save all of results and model weights for 100 epoches    
    all_result.to_csv(args.out_dirc+'all_result.csv',index=0)

    tb_writer.close()
if __name__=='__main__':
    args=model_args()
    out_dir = 'seed_{}_batchsize_{}_lr_{}_wd_{}_hyperparameters_alpha_{}_beta_{}_ST/'.format(
    args.seed, args.batch_size, args.lr, args.wd,args.loss_weight_alpha,args.loss_weight_beat
    )
    args.out_dirc = out_dir
    print(out_dir)
    os.makedirs(args.out_dirc,exist_ok=True)
    os.makedirs(args.out_dirc+'epoch_result/',exist_ok=True)
    os.makedirs(args.out_dirc+'weight/',exist_ok=True)
    main(args)