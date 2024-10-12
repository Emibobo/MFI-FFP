import torch
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
# feature extraction  branches
class feature(nn.Module):
    def __init__(self,dim=512):
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.avgpool = torch.nn.Identity() 
        self.resnet18.fc = torch.nn.Identity() 

        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.mobilenet.classifier[0] = nn.Linear(1280, 25088)  
        self.mobilenet.classifier[1] = nn.Identity()
        self.vit = models.vit_b_16(pretrained=True)
        self.vit.heads[0]=nn.Linear(768, 25088)
        
        self.IRB=self.mobilenet
        self.RB=self.resnet18
    def forward(self,x1,x2,x3):
        
        f1=self.IRB(x1).view(-1,512,7,7)#IRB Block
        f2=self.RB(x2).view(-1,512,7,7)#RB Block
        f3=self.vit(x3).view(-1,512,7,7)#ViT Block
        return f1,f2,f3

class Global_local_attention(nn.Module):
    def __init__(self,channels, r):
        super().__init__()
        inter_channels = int(channels // r)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # self.sigmoid=nn.Sigmoid()
    
    def forward(self,input):
        out_global=self.global_att(input)
        out_local=self.local_att(input)
        # wei=self.sigmoid(out_global+out_local)
        out= out_global+out_local
        return out
  
# Feature fusion block
class MFF(nn.Module):
    def __init__(self,channels=512,r=8,iteration_number=None):
        super().__init__()
        self.GLA=Global_local_attention(channels,r)
        self.GLA1=Global_local_attention(channels,r)
        self.GLA2=Global_local_attention(channels,r)
        self.iteration_number=iteration_number
        self.sigmoid=nn.Sigmoid()
        
    def forward(self,F1,F2,F3):
        F=F1+F2+F3
        out=self.GLA(F)
        wei=self.sigmoid(out)
        return F1*wei+F2*(1-wei)+F3*wei

class classifier(nn.Module):
    def __init__(self,dim=512,num_classes=2):
        super().__init__()
        self.feature_extractor=feature()
        self.feature_fusion=MFF()
        self.avg_pool=nn.AdaptiveAvgPool2d((1,1))
        self.classifier=nn.Sequential(
                    nn.Linear(dim,256),
                    nn.ReLU(),
                    nn.Dropout(p=0.3),
                    nn.Linear(256,num_classes)
                    )
        self.softmax=nn.Softmax(dim=-1)
    def forward(self,x1,x2,x3):
        #features extraction
        f1,f2,f3=self.feature_extractor(x1,x2,x3)
        #pooling layer
        out=self.avg_pool(self.feature_fusion(f1,f2,f3))
        #view->[b,536]
        out=out.view(out.size(0),512)
        #[b.536]->[b,2]
        out=self.softmax(self.classifier(out))
        return out
