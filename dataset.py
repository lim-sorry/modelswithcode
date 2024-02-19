import os
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Resize
import random

class CustomTransform(nn.Module):
    def __init__(self, height:int, width:int) -> None:
        super(type(self), self).__init__()
        self.height = height
        self.width = width
        self.to_tensor = ToTensor()
        self.resize = Resize((self.height, self.width), antialias=True)        
        
    def forward(self, x) -> torch.Tensor:
        x = self.to_tensor(x)
        x = self.resize(x)
        x = ((x / 255.0) / 0.5) - 0.5

        return x
    
class CustomDataset(Dataset):
    '''
    #### CustomDataset
    Loads image list when initialized. Load actual image datas when batch sampling to save resources.
    args
        * root_path: CelebA dir path containing ['Anno', 'Eval', 'Img/img_align_celeba'] dirs
        * disc: 'train' or 'valid' or 'test' based on 'root_path/Eval/list_eval_partition.txt'
    '''
    def __init__(self, root_celeba:str, root_rafd:str, disc:str, transform:nn.Module) -> None:
        super(type(self), self).__init__()
        self.root_celeba = root_celeba
        self.disc = {'train':0, 'valid':1, 'test':2}[disc]
        self.transform = transform
        
        self.df_celeba = self.get_images_from_celeba()[:10000]
        self.df_rafd = self.get_images_from_celeba()
        self.adjust_dataset_len()
    
    def __getitem__(self, index:int) -> tuple[dict, dict]:
        image_celeba = cv2.imread(os.path.join(self.root_celeba, 'Img/img_align_celeba', self.df_celeba.iloc[index, 0]))
        image_celeba = self.transform(image_celeba)
        label_celeba = torch.tensor(self.df_celeba.iloc[index, 1:].to_numpy(dtype=np.float32))
        celeba = {'image':image_celeba, 'label':label_celeba}

        return celeba, celeba
    
    def __len__(self) -> int:
        return len(self.df_celeba)
    
    def get_images_from_celeba(self) -> pd.DataFrame:
        disc = pd.read_csv(os.path.join(self.root_celeba,'Eval/list_eval_partition.txt'), sep=' ', names=['image', 'disc'])
        disc = disc['disc'] == self.disc

        df = pd.read_csv(os.path.join(self.root_celeba,'Anno/list_attr_celeba.txt'), sep='\s+', skiprows=2, header=None)
        df = df.loc[disc]

        return df
        
    def get_images_from_rafd(self) -> pd.DataFrame:
        pass
    
    def adjust_dataset_len(self) -> None:
        if len(self.df_celeba) < len(self.df_rafd):
            indices = np.arange(len(self.df_rafd))
            random.shuffle(indices)
            self.df_rafd = self.df_rafd.iloc[indices[:len(self.df_celeba)]].reset_index(drop=True)
        elif len(self.df_celeba) > len(self.df_rafd):
            indices = np.arange(len(self.df_celeba))
            random.shuffle(indices)
            self.df_celeba = self.df_celeba.iloc[indices[:len(self.df_rafd)]].reset_index(drop=True)
