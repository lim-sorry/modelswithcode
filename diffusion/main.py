import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import tqdm
from model import UNet
from function import DDIM
from data import CelebaDataLoader, Transformer
from torchvision.utils import save_image, make_grid
from torchvision.transforms import ToTensor

def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--PATH_IMG_FOLDER', type=str, default='/root/CelebA/Img/img_align_celeba')
    parser.add_argument('--PATH_LABEL', type=str, default='/root/CelebA/Anno/list_attr_celeba.txt')

    parser.add_argument('--IMG_HEIGHT', type=int, default=64)
    parser.add_argument('--IMG_WIDTH', type=int, default=64)
    parser.add_argument('--IMG_CH', type=int, default=3)

    parser.add_argument('--BATCH_SIZE', type=int, default=64)
    parser.add_argument('--EPOCH', type=int, default=1024)
    parser.add_argument('--LEARNING_RATE', type=float, default=1e-5)
    parser.add_argument('--WEIGHT_DECAY', type=float, default=1e-5)

    parser.add_argument('--MAX_STEP', type=int, default=256)
    parser.add_argument('--BETA_ST', type=float, default=0.0001)
    parser.add_argument('--BETA_END', type=float, default=0.02)

    parser.add_argument('--ITER', type=int, default=0)
    parser.add_argument('--PATH_CHECKPOINT', type=str, default='diffusion.pt')

    parser.add_argument('--MODE', type=str, default='train')

    return parser.parse_args('')


class Diffusion:
    def __init__(self, opt) -> None:
        self.opt = opt
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.transformer = Transformer((opt.IMG_HEIGHT, opt.IMG_WIDTH))
        self.dataloader = CelebaDataLoader(opt.PATH_IMG_FOLDER, opt.PATH_LABEL, self.transformer, opt.BATCH_SIZE)
        self.ddim = DDIM(opt.BETA_ST, opt.BETA_END, opt.MAX_STEP, self.device)
        
        self.model = UNet(opt.MAX_STEP).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), opt.LEARNING_RATE, weight_decay=opt.WEIGHT_DECAY)
        self.epoch = 0

        if os.path.exists(opt.PATH_CHECKPOINT):
            checkpoint = torch.load(opt.PATH_CHECKPOINT)
            self.epoch = checkpoint['epoch']+1
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.optimizer.param_groups[0]['lr'] = opt.LEARNING_RATE
            

    def train(self) -> None:
        opt = self.opt
        device = self.device

        for epoch in range(self.epoch, opt.EPOCH+1):
            self.model.train()
            tq = tqdm.tqdm(self.dataloader, ncols=150)
            for i, (img, label) in enumerate(tq):
                img = img.to(device)
                t = torch.randint(0, opt.MAX_STEP, (opt.BATCH_SIZE,), device=device).float()
                noise_img, noise = self.ddim.sample_noise_image(img, t)
                noise_pred = self.model(noise_img, t)
                loss = F.mse_loss(noise_pred, noise)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                tq.set_postfix_str(f'{epoch:2d}/{loss.item():>7.4f}')
            
            self.model.eval()
            with torch.no_grad():
                sample_img = torch.randn(4,3,64,64).to(device)
                imgs = [sample_img[0]]
                for i in range(0, opt.MAX_STEP)[::-1]:
                    t = torch.full((1,), i, device=self.device).float()
                    noise_pred = self.model(sample_img, t)  # Predicted noise
                    sample_img = self.ddim.denoise_image(sample_img, noise_pred, t)
                    if i % 18 == 0:
                        imgs.append(sample_img[0])
                imgs = torch.stack(imgs, dim=0)
                save_image(make_grid((imgs+1)/2, 4), f'images/train_img.png')
            self.save_checkpoint(epoch, self.model, self.optimizer, opt.PATH_CHECKPOINT)


    def test(self):
        opt = self.opt
        device = self.device
        
        sample_img = torch.randn((64, opt.IMG_CH, opt.IMG_HEIGHT, opt.IMG_WIDTH)).to(device)
        with torch.no_grad():
            for i in range(0, opt.MAX_STEP)[::-1]:
                t = torch.full((1,), i, device=self.device).float()
                noise_pred = self.model(sample_img, t)  # Predicted noise
                sample_img = self.ddim.denoise_image(sample_img, noise_pred, t)
        save_image(make_grid((sample_img+1)/2, 8), f'images/test_img.png')


    def save_checkpoint(self, epoch:int, model, optimizer, filename:str):
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filename)


def main():
    opt = parse_arg()
    diffusion = Diffusion(opt)
    
    if opt.MODE.lower() == 'train':
        diffusion.train()
    if opt.MODE.lower() == 'test':
        diffusion.test()



if __name__ == '__main__':
    main()