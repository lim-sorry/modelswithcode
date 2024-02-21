import os
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
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
        x = ((x - 0.5) / 0.5)

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
        self.root_rafd = root_rafd
        self.disc = {'train':0, 'valid':1, 'test':2}[disc]
        self.transform = transform


    def __getitem__(self, index:int) -> tuple[dict, dict]:
        if self.name_a == 'celeba':    
            image_a = cv2.imread(os.path.join(self.root_celeba, 'Img/img_align_celeba', self.df_a.iloc[index, 0]))
            image_a = cv2.cvtColor(image_a, cv2.COLOR_RGB2BGR)
            image_a = self.transform(image_a)
        label_a = torch.tensor(self.df_a.iloc[index, 1:].to_numpy(dtype=np.float32))
        dict_a = {'image':image_a, 'label':label_a}

        if self.name_b == 'celeba':    
            image_b = cv2.imread(os.path.join(self.root_celeba, 'Img/img_align_celeba', self.df_b.iloc[index, 0]))
            image_b = cv2.cvtColor(image_b, cv2.COLOR_RGB2BGR)
            image_b = self.transform(image_b)
        label_b = torch.tensor(self.df_b.iloc[index, 1:].to_numpy(dtype=np.float32))
        dict_b = {'image':image_b, 'label':label_b}

        return dict_a, dict_b
    

    def __len__(self) -> int:
        return len(self.df_a)
    

    def get_images(self, label_a:list, label_b:list, name_a:str='celeba', name_b:str='rafd') -> None:
        self.name_a = name_a
        self.name_b = name_b
        self.df_a = self._get_images_from_rafd(label_a) if name_a=='rafd' else self._get_images_from_celeba(label_a)
        self.df_b = self._get_images_from_rafd(label_b) if name_b=='rafd' else self._get_images_from_celeba(label_b)
        self._adjust_dataset_len()
    

    def _get_images_from_celeba(self, label:list) -> pd.DataFrame:
        disc = pd.read_csv(os.path.join(self.root_celeba,'Eval/list_eval_partition.txt'), sep=' ', names=['image', 'disc'])
        disc = disc['disc'] == self.disc

        df = pd.read_csv(os.path.join(self.root_celeba,'Anno/list_attr_celeba.txt'), sep='\s+', skiprows=1)
        df = df.reset_index(drop=False)
        df = df.loc[disc, ['index'] + label]

        return df
                

    def _get_images_from_rafd(self) -> pd.DataFrame:
        pass
    

    def _adjust_dataset_len(self) -> None:
        if len(self.df_a) <= len(self.df_b):
            indices = np.arange(len(self.df_b))
            random.shuffle(indices)
            self.df_b = self.df_b.iloc[indices[:len(self.df_a)]].reset_index(drop=True)
        elif len(self.df_a) > len(self.df_b):
            indices = np.arange(len(self.df_a))
            random.shuffle(indices)
            self.df_a = self.df_a.iloc[indices[:len(self.df_b)]].reset_index(drop=True)
