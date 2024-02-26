import os
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader
from dataset import CustomDataset, CustomTransform
from model import Discriminator, Generator
from criterion import calc_adversal_loss, calc_domain_cls_loss, calc_reconstruction_loss
from utils import parse_opt, save_model, save_image_grid
from torchvision.utils import save_image


class Test:
    def __init__(self, opt) -> None:
        self.opt = opt
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        os.makedirs('img', exist_ok=True)

        self.len_a = len(opt.label_a)

        self.transform = CustomTransform(opt.height, opt.width)
        self.dataset = CustomDataset(opt.root_celeba, opt.root_rafd, 'test', self.transform)
        self.dataset.get_images(opt.label_a, opt.label_b, opt.name_a, opt.name_b)

        self.gen = torch.load('gen.pt').to(self.device)
        print('gen loaded')
        self.disc = torch.load('disc.pt').to(self.device)
        print('disc loaded')
        
    def test(self):
        opt = self.opt
        device = self.device
        
        sample = self.dataset.__getitem__(opt.test_img)[1]['image'].to(device)
        sample = sample.reshape(1,3,128,128)
        print(opt.test_label)
        domain_sample = torch.tensor([float(s) for s in opt.test_label]).float().to(device)
        domain_sample = domain_sample.reshape(-1, 5)
        
        fake_sample = self.gen(sample, domain_sample)
        save_image(sample * 0.5 + 0.5, f'real_sample.png')
        save_image(fake_sample * 0.5 + 0.5, f'fake_sample.png')
        
import sys

def main():
    opt = parse_opt()

    test = Test(opt)
    test.test()
    

if __name__ == "__main__":
    main()
