from helper_functions import generate_dataset
from torchvision.io import read_image

from transformers import AutoImageProcessor
import torch 
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
from torchvision.io import ImageReadMode
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
class BiradsDataSet(Dataset):
    def __init__(self,data_dir,modality,transform=None,category="Train",normalize=True):
        super().__init__()
        self.category=category
        self.normalize = normalize
        self.transform=transform
        self.dataframe = generate_dataset(data_dir)
        self.labels = np.array(self.dataframe["Classification"].apply(lambda x:int(x[-1])))
      
        self.modality=None
        if(modality=="Mammogram"):
            self.img_dirs = np.array(self.dataframe["Mammogram_Path"] )
        elif(modality=="UltraSound"):
            self.img_dirs = np.array(self.dataframe["UltraSound_Path"]) 
        elif(modality=="Multimodal"): 
            img_dirs1 = np.array(self.dataframe["UltraSound_Path"]) 
            img_dirs2 = np.array(self.dataframe["Mammogram_Path"] )
            self.modality="Multimodal"
            self.img_dirs = np.rec.fromarrays([img_dirs1,img_dirs2])
                
        train_img_dirs,val_imgs,train_labels,val_labels = train_test_split(self.img_dirs,self.labels,test_size=0.3,stratify=self.labels,random_state=42)
     
        extra_imgs = []
        extra_labels = []
        for i,img_dir in enumerate(train_img_dirs): 
            if(train_labels[i]>3):
                for j in range(5):
                    extra_imgs.append(img_dir)
                    extra_labels.append(train_labels[i])

        # print(len(extra_imgs))
        # print(len(extra_labels))
        for img_dir in extra_imgs:
            train_img_dirs=np.append(train_img_dirs,img_dir)
        print(len(train_img_dirs)+len(val_imgs))
        for label in extra_labels:
            train_labels=np.append(train_labels,label)
        if(category=="Train"):
            self.img_dirs = train_img_dirs
            self.labels = train_labels
        if(category=="Val"):
            self.img_dirs=val_imgs
            self.labels = val_labels



    def __len__(self):
        return len(self.labels)
    def __getitem__(self,idx):
        if(self.modality!="Multimodal"):
            image_path = self.img_dirs[idx]
            label = self.labels[idx]
            image = read_image(image_path,ImageReadMode.GRAY)
            image = image.expand(3,*image.shape[1:])
            image = image_processor(images=image,return_tensors="pt")['pixel_values'][0]
            # if(self.transform):
            #     image = self.transform(image)
            # if(self.normalize):
            #     image = F.normalize(image,p=2,dim=1)
            return image,label
        else:
            image_1_path = self.img_dirs[idx][0]
            image_2_path = self.img_dirs[idx][1]
            label = self.labels[idx]
            image = read_image(image_1_path,ImageReadMode.GRAY)
            image = image.expand(3,*image.shape[1:])
            image = image_processor(images=image,return_tensors="pt")['pixel_values'][0]
            
            image2 = read_image(image_2_path,ImageReadMode.GRAY)
            image2 = image2.expand(3,*image2.shape[1:])
            image2 = image_processor(images=image2,return_tensors="pt")['pixel_values'][0]
            return image,image2,label
# class BiradsDataModule(L.LightningDataModule):
#     def __init__(self,data_dir,batch_size,num_workers,modality,transform=None):
#         super().__init__()
#         self.transform=transform
#         self.save_hyperparameters()
#         self.prepare_data_per_node=True
#         self.data_dir = data_dir
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.modality=modality
#
#     def setup(self,stage):
#         self.Train_Dataset = BiradsDataSet(self.data_dir,self.modality,transform=self.transform,category="Train")
#         self.Val_Dataset = BiradsDataSet(self.data_dir,self.modality,transform=self.transform,category="Val")
#     def train_dataloader(self):
#         return DataLoader(self.Train_Dataset,batch_size=self.batch_size,shuffle=True)
#     def val_dataloader(self):
#         return DataLoader(self.Val_Dataset,batch_size=self.batch_size)
#        
