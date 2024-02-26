import os
import random
from typing import Iterable, List
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.dataloader import _collate_fn_t, _worker_init_fn_t
import glob
from torchvision.transforms import *

class Transformer:
    def __init__(self, img_size:tuple) -> None:
        self.transformer = Compose([
            ToTensor(),
            Resize(img_size, antialias=True),
            Lambda(lambda x: (x*2) - 1)
            # Normalize(0.5, 0.5, inplace=True),
        ])

        self.augment = RandomApply([
            RandomRotation([-15.0, 15.0]),
            RandomResizedCrop(img_size, [0.8, 1.0], [1.0, 1.0], antialias=True),
        ], p=0.5)
    
    def transform(self, img, aug:bool=False):
        img = self.transformer(img)
        if aug:
            img = self.augment(img)
        return img


class CelebaDataset(Dataset):
    def __init__(self, path_img_folder:str, path_label:str, transformer:Transformer) -> None:
        super(type(self), self).__init__()

        self.list_img_path = glob.glob(os.path.join(path_img_folder, '*.jpg'))
        self.df_label = pd.read_table(path_label, sep='\s+', skiprows=1)
        self.df_label.columns = map(lambda col: col.replace('_', ' '), self.df_label.columns)
        
        self.transformer = transformer
        

    def __getitem__(self, index) -> tuple:
        img = cv2.imread(self.list_img_path[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transformer.transform(img, True)

        label = []
        for col in self.df_label.columns:
            if col == 'No Beard':
                label.append('Beard') if self.df_label.iloc[index][col] == -1 else None
            elif col == 'Male':
                label.append('Male' if self.df_label.iloc[index][col] == 1 else 'Female')
            elif col == 'Young':
                label.append('Young' if self.df_label.iloc[index][col] == 1 else 'Old')
        random.shuffle(label)
        label = ' and '.join(label)

        return img, label
    
    
    def __len__(self):
        return len(self.list_img_path)
    

    def transform(self, image):
        return self.transform(image)


class CelebaDataLoader(DataLoader):
    def __init__(self, path_img_folder:str, path_label:str, transformer:Transformer, batch_size:int, shuffle:bool=True, drop_last:bool=True):
        super(type(self), self).__init__(dataset=CelebaDataset(path_img_folder, path_label, transformer), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
