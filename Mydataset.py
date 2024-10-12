from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import torchvision.transforms as transforms
import numpy as np
class MyDataSet(Dataset):
    '''define a dataset for your own data '''
    def __init__(self,args,data_x,data_y, transform=None):
        super().__init__()
        self.data_csv_rgb=pd.read_csv(args.data_path_rgb)
        self.data_csv_icg=pd.read_csv(args.data_path_icg)
        self.data_csv_color=pd.read_csv(args.data_path_color)
        self.transform=transform
        samples=[]
        for i in range(0,len(data_x)):
            try:
                res_rgb=self.data_csv_rgb[self.data_csv_rgb['img'].str.contains(data_x.iloc[i])]
                res_icg=self.data_csv_icg[self.data_csv_icg['img'].str.contains(data_x.iloc[i])]
                res_color=self.data_csv_color[self.data_csv_color['img'].str.contains(data_x.iloc[i])]
                data_path_rgb=res_rgb['path'].iloc[0]
                data_path_icg=res_icg['path'].iloc[0]
                data_path_color=res_color['path'].iloc[0]
                img_rgb=Image.open(data_path_rgb).convert('RGB')
                img_icg=Image.open(data_path_icg).convert('RGB')
                img_color=Image.open(data_path_color).convert('RGB')
                label=data_y.iloc[i]
                Tri_modality_sample={
                    'name':data_x.iloc[i],
                    'img_rgb':img_rgb,
                    'img_icg':img_icg,
                    'img_color':img_color,
                    'label':label
                }
                samples.append(Tri_modality_sample)
            except:
                continue
        self.samples=samples
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, item):
        sample=self.samples[item]
        if self.transform is not None:
                sample['img_rgb']=self.transform(sample['img_rgb'])
                sample['img_icg']=self.transform(sample['img_icg'])
                sample['img_color']=self.transform(sample['img_color'])
        return sample

def get_data_loader(args,data_x,data_y):
    transformer=transforms.Compose([
        #imagesize->[224,224]
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomRotation(20),
        transforms.ToTensor()
        ])
    #Intializing the dataset
    dataset=MyDataSet(args, data_x,data_y, transform=transformer)
    #Using DataLoader to load the data
    data_loader=DataLoader(dataset, batch_size=args.batch_size,shuffle=True, num_workers=args.num_workers,pin_memory=True)
    return data_loader

class ValDataSet(Dataset):
    '''define a dataset for your own data '''
    def __init__(self,args,data_x,data_y, transform=None):
        super().__init__()
        self.data_csv_rgb=pd.read_csv(args.val_rgb)
        self.data_csv_icg=pd.read_csv(args.val_icg)
        self.data_csv_color=pd.read_csv(args.val_color)
        self.transform=transform
        samples=[]
        for i in range(0,len(data_x)):
            try:
                res_rgb=self.data_csv_rgb[self.data_csv_rgb['img'].str.contains(data_x.iloc[i])]
                res_icg=self.data_csv_icg[self.data_csv_icg['img'].str.contains(data_x.iloc[i])]
                res_color=self.data_csv_color[self.data_csv_color['img'].str.contains(data_x.iloc[i])]
                data_path_rgb=res_rgb['path'].iloc[0]
                data_path_icg=res_icg['path'].iloc[0]
                data_path_color=res_color['path'].iloc[0]
                img_rgb=Image.open(data_path_rgb).convert('RGB')
                img_icg=Image.open(data_path_icg).convert('RGB')
                img_color=Image.open(data_path_color).convert('RGB')
                label=data_y.iloc[i]
                Tri_modality_sample={
                    'name':data_x.iloc[i],
                    'img_rgb':img_rgb,
                    'img_icg':img_icg,
                    'img_color':img_color,
                    'label':label
                }
                samples.append(Tri_modality_sample)
            except:
                continue
            # print(data_x.iloc[i],np.array(img_rgb).shape,np.array(img_icg).shape)
        self.samples=samples
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, item):
        sample=self.samples[item]
        # if sample['img_rgb'] !='RGB' :
        #     raise ValueError("image:{}_RGB isn't RGB mode.".format(sample['name']))    
        # elif sample['img_icg'] !='RGB':
        #     raise ValueError("image:{}_ICG isn't RGB mode.".format(sample['name']))    
        if self.transform is not None:
                sample['img_rgb']=self.transform(sample['img_rgb'])
                sample['img_icg']=self.transform(sample['img_icg'])
                sample['img_color']=self.transform(sample['img_color'])
        return sample
   
def val_data_loader(args,data_x,data_y):
    transformer=transforms.Compose([
        #imagesize->[224,224]
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomRotation(20),
        transforms.ToTensor()
        ])
    #Intializing the dataset
    dataset=ValDataSet(args, data_x,data_y, transform=transformer)
    #Using DataLoader to load the data
    data_loader=DataLoader(dataset, batch_size=args.batch_size,shuffle=True, num_workers=args.num_workers,pin_memory=True)
    return data_loader    
        