import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import *

class Transform(nn.Module):
    def __init__(self, img_size:int) -> None:
        super(type(self), self).__init__()
        self.img_size = img_size

        self.transform = Compose([
            PILToTensor(),
            Resize((img_size, img_size), antialias=True),
        ])
        self.augment = Compose([
            RandomApply([RandomResizedCrop((img_size, img_size), (0.8, 1.0), antialias=True)], 0.5),            
            RandomHorizontalFlip(0.5),
            # RandomAutocontrast(0.1),
            # RandomGrayscale(0.1),
        ])

    def forward(self, img, trg) -> list[Tensor, Tensor]:
        img = self.transform(img)
        trg = self.transform(trg)
        tmp = torch.cat([img, trg], dim=0)
        tmp = self.augment(tmp)
        
        img = tmp[:-1] / 255.0
        img = (img - .5) / 0.5

        trg = tmp[-1] - 1.0
        # trg = trg.repeat(3,1,1)
        trg = nn.functional.one_hot(trg.type(torch.int64), 3).float()
        trg = trg.reshape((self.img_size, self.img_size, 3)).permute(2,0,1)

        return img, trg
    
if __name__ == '__main__':
    transform = Transform(256)
    dataset = OxfordIIITPet('~/data', 'trainval', 'segmentation', transform)
    dataloader = DataLoader(dataset, 64, True, drop_last=True)
    for img, trg in dataloader:
        print(img[0][0])
        break