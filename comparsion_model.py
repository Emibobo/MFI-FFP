import torch
import torchvision.models as models
import torch.nn as nn

class comparsion_models(nn.Module):
    def __init__(self,dim=512,num_classes=2,name='swin_s'):
        super().__init__()
        self.num_classes=num_classes
        self.name=name
        self.swin_s = models.swin_s(pretrained=True)
        self.swin_s.head=nn.Linear(768, dim)
        self.efficientnet=models.efficientnet_v2_s(pretrained=True)
        self.efficientnet.classifier[-1]=nn.Linear(1280, dim)
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.fc = torch.nn.Identity() 
        self.convnext=models.convnext_small(pretrained=True)
        self.convnext.classifier[-1]=nn.Linear(768,dim)
        self.mobilenet=models.mobilenet_v3_small(pretrained=True)
        self.mobilenet.classifier[-1]=nn.Linear(1024,dim)
    def forward(self, x1,x2,x3):
        if self.name =='swin_s':
            f1= self.swin_s(x1).view(-1,512,1,1)
            f2= self.swin_s(x2).view(-1,512,1,1)
            f3= self.swin_s(x3).view(-1,512,1,1)
        elif self.name =='efficientnet':
            f1= self.efficientnet(x1).view(-1,512,1,1)
            f2= self.efficientnet(x2).view(-1,512,1,1)
            f3= self.efficientnet(x3).view(-1,512,1,1)
        elif self.name =='resnet18':
            f1= self.resnet18(x1).view(-1,512,1,1)
            f2= self.resnet18(x2).view(-1,512,1,1)
            f3= self.resnet18(x3).view(-1,512,1,1)
        elif self.name =='convnext':
            f1= self.convnext(x1).view(-1,512,1,1)
            f2= self.convnext(x2).view(-1,512,1,1)
            f3= self.convnext(x3).view(-1,512,1,1)
        elif self.name =='mobilenet':
            f1= self.mobilenet(x1).view(-1,512,1,1)
            f2= self.mobilenet(x2).view(-1,512,1,1)
            f3= self.mobilenet(x3).view(-1,512,1,1)
        else:
            raise ValueError('No such model')
        f = torch.cat((f1, f2, f3), dim=1)
        return f

class comparsion_models_no_concat(nn.Module):
    def __init__(self,dim=512,num_classes=2,name='swin_s'):
        super().__init__()
        self.num_classes=num_classes
        self.name=name
        self.swin_s = models.swin_s(pretrained=True)
        self.swin_s.head=nn.Linear(768, dim)
        self.efficientnet=models.efficientnet_v2_s(pretrained=True)
        self.efficientnet.classifier[-1]=nn.Linear(1280, dim)
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.fc = torch.nn.Identity() 
        self.convnext=models.convnext_small(pretrained=True)
        self.convnext.classifier[-1]=nn.Linear(768,dim)
        self.mobilenet=models.mobilenet_v3_small(pretrained=True)
        self.mobilenet.classifier[-1]=nn.Linear(1024,dim)
    def forward(self, x1,x2,x3):
        if self.name =='swin_s':
            f1= self.swin_s(x1).view(-1,512,1,1)
            f2= self.swin_s(x2).view(-1,512,1,1)
            f3= self.swin_s(x3).view(-1,512,1,1)
        elif self.name =='efficientnet':
            f1= self.efficientnet(x1).view(-1,512,1,1)
            f2= self.efficientnet(x2).view(-1,512,1,1)
            f3= self.efficientnet(x3).view(-1,512,1,1)
        elif self.name =='resnet18':
            f1= self.resnet18(x1).view(-1,512,1,1)
            f2= self.resnet18(x2).view(-1,512,1,1)
            f3= self.resnet18(x3).view(-1,512,1,1)
        elif self.name =='convnext':
            f1= self.convnext(x1).view(-1,512,1,1)
            f2= self.convnext(x2).view(-1,512,1,1)
            f3= self.convnext(x3).view(-1,512,1,1)
        elif self.name =='mobilenet':
            f1= self.mobilenet(x1).view(-1,512,1,1)
            f2= self.mobilenet(x2).view(-1,512,1,1)
            f3= self.mobilenet(x3).view(-1,512,1,1)
        else:
            raise ValueError('No such model')
        return f1,f2,f3

def qkv_fusion(tensor_q, tensor_k, tensor_v):
    # Calculate the attention weights
    attention_weights = torch.matmul(tensor_q, tensor_k.transpose(-2, -1))
    attention_weights = torch.softmax(attention_weights, dim=-1)

    # Fuse the feature tensors using the attention weights
    fused_tensor = torch.matmul(attention_weights, tensor_v)

    return fused_tensor

# Example usage
tensor_q = torch.randn(1, 512, 1,1)  # Q tensor with shape (batch_size, sequence_length, hidden_size)
tensor_k = torch.randn(1, 512, 1,1)  # K tensor with shape (batch_size, sequence_length, hidden_size)
tensor_v = torch.randn(1, 512, 1,1)  # V tensor with shape (batch_size, sequence_length, hidden_size)

fused_tensor = qkv_fusion(tensor_q, tensor_k, tensor_v)
print(fused_tensor.shape)  # Output shape: (batch_size, sequence_length, hidden_size)

#Comparision models with concatenation mode
class classifier_comparision_Concat(nn.Module):
    def __init__(self,dim=1536,num_classes=2,name=None):
        super().__init__()
        self.feature_extractor=comparsion_models(name=name)
        self.avg_pool=nn.AdaptiveAvgPool2d((1,1))
        self.classifier=nn.Sequential(
                    nn.Linear(dim,512),
                    nn.ReLU(),
                    nn.Linear(512,256),
                    nn.ReLU(),
                    nn.Dropout(p=0.3),
                    nn.Linear(256,num_classes)
                    )
        self.softmax=nn.Softmax(dim=-1)
    def forward(self,x1,x2,x3):
        #features extraction
        f=self.feature_extractor(x1,x2,x3)
        #pooling layer
        out=self.avg_pool(f)
        #view->[b,536]
        out=out.view(out.size(0),1536)
        #[b.536]->[b,2]
        out=self.softmax(self.classifier(out))
        return out

#Comparision models with Attention mode
class classifier_comparision_QKV(nn.Module):
    def __init__(self,dim=512,num_classes=2,name=None):
        super().__init__()
        self.feature_extractor=comparsion_models_no_concat(name=name)
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
        F=qkv_fusion(f1,f2,f3)
        #pooling layer
        out=self.avg_pool(F)
        #view->[b,536]
        out=out.view(out.size(0),512)
        #[b.536]->[b,2]
        out=self.softmax(self.classifier(out))
        return out