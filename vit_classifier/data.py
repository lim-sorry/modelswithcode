import os
import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import glob
from torchvision.transforms import *

class Transformer:
    def __init__(self, img_size:tuple) -> None:
        self.transformer = Compose([
            ToTensor(),
            Resize(img_size, antialias=True),
            Normalize(0.5, 0.5, inplace=True)
        ])

        self.augment = RandomApply([
            RandomRotation([-15.0, 15.0]),
            RandomResizedCrop(img_size, [0.8, 1.0], [4/5, 5/4], antialias=True),
        ], p=0.5)
    
    def transform(self, img, aug:bool=False):
        img = self.transformer(img)
        if aug:
            img = self.augment(img)
        return img


class CelebaDataset(Dataset):
    def __init__(self, path_img:str, path_label:str, transformer:Transformer, train:bool) -> None:
        super(type(self), self).__init__()

        self.list_img_path = glob.glob(os.path.join(path_img, '*.jpg'))
        self.df_label = pd.read_table(path_label, sep='\s+', skiprows=1)
        
        self.transformer = transformer

        slice_idx = int(len(self.df_label) * 0.9)
        if train:
            self.list_img_path = self.list_img_path[:slice_idx]
            self.df_label = self.df_label[:slice_idx]
        else:
            self.list_img_path = self.list_img_path[slice_idx:]
            self.df_label = self.df_label[slice_idx:]


    def __getitem__(self, index) -> tuple:
        img = cv2.imread(self.list_img_path[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transformer.transform(img, True)

        label = 1 if self.df_label.iloc[index]['Male'] == 1 else 0

        return img, label
    
    
    def __len__(self):
        return len(self.list_img_path)
    

    def transform(self, image):
        return self.transform(image)


class CelebaDataLoader(DataLoader):
    def __init__(self, path_img:str, path_label:str, transformer:Transformer, train:bool, batch_size:int, shuffle:bool=True, drop_last:bool=True):
        super(type(self), self).__init__(dataset=CelebaDataset(path_img, path_label, transformer, train), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
