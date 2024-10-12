import torch
import torch.nn as nn
from tqdm import tqdm
import sys
from sklearn.metrics import accuracy_score, roc_auc_score,confusion_matrix,precision_score, recall_score, f1_score
import numpy as np
import torch.nn.functional as F
import pandas as pd
import torch
from torch import nn
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Focal loss 
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.85, epsilon=1.e-9):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, input, target):
        """
        Args:
            input: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        multi_hot_key = target
        logits = input[:,1]
        zero_hot_key = 1 - multi_hot_key
        loss = -self.alpha * multi_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
        loss += -(1 - self.alpha) * zero_hot_key * torch.pow(logits, self.gamma) * (1 - logits + self.epsilon).log()
        return loss.mean()

#intra-class loss
class intraclass_loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,input,target):
        class_0_indices = torch.where(target == 0)[0]
        class_1_indices = torch.where(target == 1)[0]
        class_0_corr=[]
        class_1_corr=[]
        for i in range(0,len(class_0_indices)):
            if i < len(class_0_indices)-1:
                tensor1=input[class_0_indices[i]]
                tensor2=input[class_0_indices[i+1]]
                covariance_0 = torch.sum((tensor1 - torch.mean(tensor1)) * (tensor2 - torch.mean(tensor2)))
                std_dev1 = torch.std(tensor1)
                std_dev2 = torch.std(tensor2)
                correlation_coefficient_0 = covariance_0 / (std_dev1 * std_dev2)
                class_0_corr.append(correlation_coefficient_0)
            
        for i in range(0,len(class_1_indices)):
            if i < len(class_1_indices)-1:
                tensor3=input[class_1_indices[i]]
                tensor4=input[class_1_indices[i+1]]
                covariance_1 = torch.sum((tensor3 - torch.mean(tensor3)) * (tensor4 - torch.mean(tensor4)))
                std_dev3 = torch.std(tensor3)
                std_dev4 = torch.std(tensor4)
                correlation_coefficient_1 = covariance_1 / (std_dev3 * std_dev4)
                class_1_corr.append(correlation_coefficient_1)
        
        if len(class_0_corr)==0:
                p_0=0
        else:
            p_0=torch.stack(class_0_corr).sum()/len(class_0_indices)
        
        if len(class_1_corr)==0:
                p_1=0
        else:
            p_1=torch.stack(class_1_corr).sum()/len(class_1_indices)
        
        loss_intra=torch.norm(p_0-p_1, p=2)**2
        return loss_intra.mean()

#LDAM loss    
class LDAMLoss(nn.Module):
 
    def __init__(self, cls_num_list=[85,900], max_m=0.5, weight=torch.tensor([0.15,0.85]).to(device), s=30):
        """
        max_m: The appropriate value for max_m depends on the specific dataset and the severity of the class imbalance.
        You can start with a small value and gradually increase it to observe the impact on the model's performance.
        If the model struggles with class separation or experiences underfitting, increasing max_m might help. However,
        be cautious not to set it too high, as it can cause overfitting or make the model too conservative.
        s: The choice of s depends on the desired scale of the logits and the specific requirements of your problem.
        It can be used to adjust the balance between the margin and the original logits. A larger s value amplifies
        the impact of the logits and can be useful when dealing with highly imbalanced datasets.
        You can experiment with different values of s to find the one that works best for your dataset and model.
        """
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list).to(device)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight
 
    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8).to(device)
        index.scatter_(1, target.detach().view(-1, 1), 1)
 
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
 
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)

def WeightedCrossEntropyLoss(input,target,weight=None,reduction='mean'):

    loss = F.cross_entropy(input, target, weight=weight, reduction=reduction)
    return loss

#the final loss function
class Customized_Loss(nn.Module):
    def __init__(self,args,epsilon=1.e-9):
        super().__init__()
        self.alpha=args.loss_weight_alpha
        self.beta=args.loss_weight_beta
        self.epsilon = epsilon
        self.LDAMLoss=LDAMLoss()
        self.Focalloss=BinaryFocalLoss()
        self.intra_loss=intraclass_loss()  
    def forward(self, input, target): 
        loss_intra=self.alpha*self.intra_loss(input, target)
        LDAM_loss=self.LDAMLoss(input, target)
        Focal_loss=self.beta*self.Focalloss(input, target)
        return LDAM_loss+Focal_loss+loss_intra

#Metrics
def ConfusionResult(label_list,pre_list,pre_prob_list):
    acc=accuracy_score(label_list,pre_list)
    auc=roc_auc_score(label_list,pre_prob_list)
    matrix=confusion_matrix(label_list,pre_list)
    tn_sum=matrix[0][0] #True Negative
    tp_sum=matrix[1][1] #False Negative
    fp_sum=matrix[0][1] #True Positive
    fn_sum=matrix[1][0] #False Positive
    Condition_negative1 = tp_sum + fn_sum + 1e-6
    Condition_negative2 = tn_sum + fp_sum + 1e-6
    prec = precision_score(label_list, pre_list,zero_division=1)
    f1 = f1_score(label_list, pre_list)
    sen=tp_sum/Condition_negative1
    spe=tn_sum/Condition_negative2
    return acc,auc,sen,spe,prec,f1

def train_one_epoch(model, criterion, data_loader, optimizer, writer,epoch):
    model.train()
    loss_function=criterion
    accu_num=torch.zeros(1).to(device)#Cumulative loss
    total_loss=0
    img_list = []
    pre_list = torch.Tensor([])
    pre_prob_list = torch.Tensor([])
    label_list = torch.Tensor([])
    epo_result = pd.DataFrame()
    sample_num=0
    data_loader=tqdm(data_loader,file=sys.stdout)
    optimizer.zero_grad()
    for step,data in enumerate(data_loader):
        input_rgbs=data['img_rgb'].to(device)
        input_icgs=data['img_icg'].to(device)
        input_colors=data['img_color'].to(device)
        label=data['label'].to(torch.int64).to(device)
        img=data['name']
        sample_num+=input_rgbs.shape[0]
        pre_prob=model(input_rgbs,input_icgs,input_colors)
        pre_class=torch.max(pre_prob,dim=1)[1]
        accu_num+=torch.eq(pre_class,label).sum()
        #loss
        loss=loss_function(pre_prob,label)
        loss.backward()
        total_loss+=loss.item()
        
        img_list.extend(img)
        label_list = torch.cat((label_list, label.int().cpu()), 0)
        pre_prob_list = torch.cat((pre_prob_list, pre_prob[:,1].cpu()), 0)
        pre_list = torch.cat((pre_list, pre_class.cpu()), 0)
        
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss,ending training',loss)
            sys.exit(1)    
        optimizer.step()
        optimizer.zero_grad()
    epo_result['img'] = img_list
    epo_result['label'] = np.array(label_list,dtype=int)
    epo_result['prob'] = np.array(pre_prob_list.detach())
    epo_result['pred'] = np.array(pre_list.detach(),dtype=int)
    epo_result['set'] = 'train'    
    acc,auc,sen,spe,prec,f1 = ConfusionResult(np.array(label_list,dtype=int),np.array(pre_list.detach(),dtype=int),np.array(pre_prob_list.detach()))
    writer.add_scalar('train/auc', auc, epoch)
    writer.add_scalar('train/acc', acc, epoch)
    writer.add_scalar('train/sen', sen, epoch)
    writer.add_scalar('train/spe', spe, epoch)
    writer.add_scalar('train/prec', prec, epoch)
    writer.add_scalar('train/f1', f1, epoch)
    
    return epo_result, total_loss, acc, auc, sen, spe, prec, f1

@torch.no_grad()
def test_one_epoch(model, criterion, data_loader, writer ,epoch):
    loss_function=criterion
    model.eval()
    img_list = []
    pre_list = torch.Tensor([])
    pre_prob_list = torch.Tensor([])
    label_list = torch.Tensor([])
    epo_result = pd.DataFrame()
  
    accu_num=torch.zeros(1).to(device)#Cumulative correct sample size
    # accu_loss=torch.zeros(1).cuda()#Cumulative loss
    total_loss=0
    sample_num=0
    data_loader=tqdm(data_loader,file=sys.stdout)
    for step,data in enumerate(data_loader):
        input_rgbs=data['img_rgb'].to(device)
        input_icgs=data['img_icg'].to(device)
        input_colors=data['img_color'].to(device)
        label=data['label'].to(torch.int64).to(device)
        sample_num+=input_rgbs.shape[0]
        img=data['name']
        pre_prob=model(input_rgbs,input_icgs,input_colors)
        pre_classes=torch.max(pre_prob,dim=1)[1]
        accu_num+=torch.eq(pre_classes,label).sum()
        
        loss=loss_function(pre_prob,label)
        total_loss+=loss.item()
        
        img_list.extend(img)
        label_list = torch.cat((label_list, label.int().cpu()), 0)
        pre_prob_list = torch.cat((pre_prob_list, pre_prob[:,1].cpu()), 0)
        pre_list = torch.cat((pre_list, pre_classes.cpu()), 0)
        # data_loader.desc="[test epoch{}] loss:{:.3f}, acc:{:.3f}".format(epoch,accu_loss/(step+1),accu_num.item()/sample_num)
    acc,auc,sen,spe,prec,f1  = ConfusionResult(label_list,pre_list,pre_prob_list)
    epo_result['img'] = img_list
    epo_result['label'] = np.array(label_list,dtype=int)
    epo_result['prob'] = np.array(pre_prob_list)
    epo_result['pred'] = np.array(pre_list,dtype=int)
    epo_result['set'] = 'test'
    writer.add_scalar('test/auc', auc, epoch)
    writer.add_scalar('test/acc', acc, epoch)
    writer.add_scalar('test/sen', sen, epoch)
    writer.add_scalar('test/spe', spe, epoch)
    writer.add_scalar('test/prec', prec, epoch)
    writer.add_scalar('test/f1', f1, epoch)
    
    return epo_result, total_loss, acc, auc, sen, spe, prec, f1

@torch.no_grad()
def valid_one_epoch(model, criterion, data_loader, writer ,epoch):
    loss_function=criterion
    model.eval()
    img_list = []
    pre_list = torch.Tensor([])
    pre_prob_list = torch.Tensor([])
    label_list = torch.Tensor([])
    epo_result = pd.DataFrame()
  
    accu_num=torch.zeros(1).to(device)#Cumulative correct sample size
    # accu_loss=torch.zeros(1).cuda()#Cumulative loss
    total_loss=0
    sample_num=0
    data_loader=tqdm(data_loader,file=sys.stdout)
    for step,data in enumerate(data_loader):
        input_rgbs=data['img_rgb'].to(device)
        input_icgs=data['img_icg'].to(device)
        input_colors=data['img_color'].to(device)
        label=data['label'].to(torch.int64).to(device)
        sample_num+=input_rgbs.shape[0]
        img=data['name']
        pre_prob=model(input_rgbs,input_icgs,input_colors)
        pre_classes=torch.max(pre_prob,dim=1)[1]
        accu_num+=torch.eq(pre_classes,label).sum()
        
        loss=loss_function(pre_prob,label)
        total_loss+=loss.item()
        
        img_list.extend(img)
        label_list = torch.cat((label_list, label.int().cpu()), 0)
        pre_prob_list = torch.cat((pre_prob_list, pre_prob[:,1].cpu()), 0)
        pre_list = torch.cat((pre_list, pre_classes.cpu()), 0)
        # data_loader.desc="[test epoch{}] loss:{:.3f}, acc:{:.3f}".format(epoch,accu_loss/(step+1),accu_num.item()/sample_num)
    acc,auc, sen, spe, prec, f1 = ConfusionResult(label_list,pre_list,pre_prob_list)
    epo_result['img'] = img_list
    epo_result['label'] = np.array(label_list,dtype=int)
    epo_result['prob'] = np.array(pre_prob_list)
    epo_result['pred'] = np.array(pre_list,dtype=int)
    epo_result['set'] = 'val'
    writer.add_scalar('val/auc', auc, epoch)
    writer.add_scalar('val/acc', acc, epoch)
    writer.add_scalar('val/sen', sen, epoch)
    writer.add_scalar('val/spe', spe, epoch)
    writer.add_scalar('val/prec', prec, epoch)
    writer.add_scalar('val/f1', f1, epoch)
    
    return epo_result, total_loss, acc, auc, sen, spe, prec, f1



def weights_init(net, init_type='normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:  
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier': 
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming': 
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':  
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0) 
        elif classname.find('Linear') != -1:
            nn.init.trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif classname.find('LayerNorm') != -1:
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)  
            
    print('initialize network with %s type' % init_type)
    net.apply(init_func)  

       
    
