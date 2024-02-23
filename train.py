import os
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader
from dataset import CustomDataset, CustomTransform
from model import Discriminator, Generator
from criterion import calc_adversal_loss, calc_domain_cls_loss, calc_reconstruction_loss
from utils import parse_opt, save_model, save_image_grid

class Train:
    def __init__(self, opt) -> None:
        self.opt = opt
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        os.makedirs('img', exist_ok=True)

        self.len_a = len(opt.label_a)
        self.n_domain = self.len_a + len(opt.label_b)

        self.transform = CustomTransform(opt.height, opt.width)
        self.dataset = CustomDataset(opt.root_celeba, opt.root_rafd, 'train', self.transform)
        self.dataset.get_images(opt.label_a, opt.label_b, opt.name_a, opt.name_b)
        self.dataloader = DataLoader(self.dataset, opt.batch_size, shuffle=True, drop_last=True, num_workers=4)

        self.t_dataset = CustomDataset(opt.root_celeba, opt.root_rafd, 'test', self.transform)
        self.t_dataset.get_images(opt.label_a, opt.label_b, opt.name_a, opt.name_b)

        if os.path.exists('gen.pt'):
            self.gen = torch.load('gen.pt').to(self.device)
            print('gen loaded')
        else:
            self.gen = Generator(opt.channel+self.n_domain).to(self.device)
            print('gen created')
        if os.path.exists('disc.pt'):
            self.disc = torch.load('disc.pt').to(self.device)
            print('disc loaded')
        else:
            self.disc = Discriminator(opt.channel, opt.height, opt.width, self.n_domain).to(self.device)
            print('disc created')
        
        self.optimizer_gen = torch.optim.Adam(self.gen.parameters(), opt.lr, weight_decay=opt.weight_decay)
        self.optimizer_disc = torch.optim.Adam(self.disc.parameters(), opt.lr, weight_decay=opt.weight_decay)
    
        self.it = opt.iter
        
        
    def train(self):
        opt = self.opt
        device = self.device

        real_sample = torch.stack([self.t_dataset.__getitem__(i)[0]['image'] for i in range(128,192)], dim=0).to(device)
        fake_label_sample_a = - torch.stack([self.t_dataset.__getitem__(i)[0]['label'] for i in range(128,192)], dim=0).to(device)
        fake_label_sample_b = - torch.stack([self.t_dataset.__getitem__(i)[1]['label'] for i in range(128,192)], dim=0).to(device)
        fake_domain_sample = torch.cat([fake_label_sample_a, fake_label_sample_b], dim=1)
        save_image_grid(real_sample, f'real.png', 8)
        
        total_loss = [0, 0, 0, 0, 0]
        for _ in range(opt.epoch):
            for dict_a, dict_b in self.dataloader:
                real_a = dict_a['image'].to(device)
                real_b = dict_b['image'].to(device)
                real_label_a = dict_a['label'].to(device)
                real_label_b = dict_b['label'].to(device)

                # 1.training discriminator
                # target label
                fake_label_a = nn.functional.one_hot(torch.randint(0, 3, real_label_a.shape[:1]), 3).to(device)
                fake_label_b = - real_label_b
                mask_a = torch.zeros_like(real_label_a)
                mask_b = torch.zeros_like(real_label_b)
                # domain label
                fake_domain_a = torch.cat([fake_label_a, mask_b], dim=1)
                fake_domain_b = torch.cat([mask_a, fake_label_b], dim=1)
                real_domain_a = torch.cat([real_label_a, mask_b], dim=1)
                real_domain_b = torch.cat([mask_a, real_label_b], dim=1)
                # fake from real and target domain
                fake_a = self.gen(real_a, fake_domain_a)
                fake_b = self.gen(real_b, fake_domain_b)
                # output from disc
                src_real_a, cls_real_a = self.disc(real_a)
                src_real_b, cls_real_b = self.disc(real_b)
                src_fake_a, _ = self.disc(fake_a)
                src_fake_b, _ = self.disc(fake_b)
                # calc adv loss
                adv_loss_a = calc_adversal_loss(src_real_a, 1) + calc_adversal_loss(src_fake_a, 0)
                adv_loss_b = calc_adversal_loss(src_real_b, 1) + calc_adversal_loss(src_fake_b, 0)
                adv_loss = adv_loss_a + adv_loss_b
                total_loss[0] += adv_loss.item()
                # reshape and apply mask
                cls_real_a = cls_real_a.view(cls_real_a.shape[:2])
                cls_real_b = cls_real_b.view(cls_real_b.shape[:2])
                cls_real_a[:,self.len_a:] = torch.zeros_like(cls_real_a[:,self.len_a:])
                cls_real_b[:,:self.len_a] = torch.zeros_like(cls_real_b[:,:self.len_a])
                # calc cls loss
                cls_loss_a = calc_domain_cls_loss(cls_real_a, real_domain_a)
                cls_loss_b = calc_domain_cls_loss(cls_real_b, real_domain_b)
                cls_loss = cls_loss_a + cls_loss_b
                total_loss[1] += cls_loss.item()
                # optimizer_disc step
                disc_loss = adv_loss + self.opt.lambda_cls * cls_loss
                self.optimizer_disc.zero_grad()
                disc_loss.backward()
                self.optimizer_disc.step()
                
                # 2.training generator
                # fake from real and target domain
                fake_a = self.gen(real_a, fake_domain_a)
                fake_b = self.gen(real_b, fake_domain_b)
                # output from disc
                src_real_a, cls_real_a = self.disc(real_a)
                src_real_b, cls_real_b = self.disc(real_b)
                src_fake_a, cls_fake_a = self.disc(fake_a)
                src_fake_b, cls_fake_b = self.disc(fake_b)
                # calc adv loss
                adv_loss_a = calc_adversal_loss(src_real_a, 0) + calc_adversal_loss(src_fake_a, 1)
                adv_loss_b = calc_adversal_loss(src_real_b, 0) + calc_adversal_loss(src_fake_b, 1)
                adv_loss = adv_loss_a + adv_loss_b
                total_loss[2] += adv_loss.item()
                # apply mask and reshape
                cls_fake_a = cls_fake_a.view(cls_fake_a.shape[:2])
                cls_fake_b = cls_fake_b.view(cls_fake_b.shape[:2])
                cls_fake_a[:,self.len_a:] = torch.zeros_like(cls_fake_a[:,self.len_a:])
                cls_fake_b[:,:self.len_a] = torch.zeros_like(cls_fake_b[:,:self.len_a])
                # calc cls loss
                cls_loss_a = calc_domain_cls_loss(cls_fake_a, fake_domain_a)
                cls_loss_b = calc_domain_cls_loss(cls_fake_b, fake_domain_b)
                cls_loss = cls_loss_a + cls_loss_b
                total_loss[3] += cls_loss.item()
                # reconstruct image
                rec_a = self.gen(fake_a, real_domain_a)
                rec_b = self.gen(fake_b, real_domain_b)
                # calc rec loss
                rec_loss_a = calc_reconstruction_loss(real_a, rec_a)
                rec_loss_b = calc_reconstruction_loss(real_b, rec_b)
                rec_loss = rec_loss_a + rec_loss_b
                total_loss[4] += rec_loss.item()
                # optimizer_gen step
                gen_loss = adv_loss + self.opt.lambda_cls * cls_loss + opt.lambda_rec * rec_loss
                self.optimizer_gen.zero_grad()
                gen_loss.backward()
                self.optimizer_gen.step()

                if self.it % 10 == 0:
                    print(self.it, total_loss)
                    total_loss = [0, 0, 0, 0, 0]
                    
                if self.it % 100 == 0:
                    with torch.no_grad():
                        fake_sample = self.gen(real_sample, fake_domain_sample)
                        save_image_grid(fake_sample, f'fake.png', 8)
                        save_image_grid(fake_sample, f'img/{int(self.it):08d}.png', 8)
                        save_model(self.gen, self.disc)
                self.it += 1
                

def main():
    opt = parse_opt()

    train = Train(opt)
    train.train()
    

if __name__ == "__main__":
    main()
