import torch
import utils
from torch.utils.data import DataLoader
from dataset import CustomDataset, CustomTransform
from model import Discriminator, Generator
from criterion import *
import tqdm
from torchvision.utils import save_image

class Train:
    def __init__(self, opt) -> None:
        self.opt = opt
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.len_a = len(opt.label_a)
        self.n_domain = self.len_a + len(opt.label_b)

        self.transform = CustomTransform(opt.height, opt.width)
        self.dataset = CustomDataset(opt.root_celeba, opt.root_rafd, 'train', self.transform)
        self.dataset.get_images(opt.label_a, opt.label_b, opt.name_a, opt.name_b)
        self.dataloader = DataLoader(self.dataset, opt.batch_size, shuffle=True, drop_last=True, num_workers=4)

        self.gen = Generator(opt.channel+self.n_domain).to(self.device)
        self.disc = Discriminator(opt.channel, opt.height, opt.width, self.n_domain).to(self.device)
        # self.gen = torch.load('gen.pt').to(self.device)
        # self.disc = torch.load('disc.pt').to(self.device)

        self.optimizer_gen = torch.optim.Adam(self.gen.parameters(), 0.0001)
        self.optimizer_disc = torch.optim.Adam(self.disc.parameters(), 0.0001)
    
        # self.tq = tqdm.tqdm(range(opt.epoch))

    def train(self):
        history = {'disc_loss':[], 'gen_loss':[]}
        #for epoch in self.tq:
        for epoch in range(self.opt.epoch):
            total_loss = [0, 0, 0, 0, 0]
            for dict_a, dict_b in self.dataloader:
                real_a = dict_a['image'].to(self.device)
                real_b = dict_b['image'].to(self.device)
                real_label_a = dict_a['label'].to(self.device)
                real_label_b = dict_b['label'].to(self.device)

                # 1.training discriminator
                # target label
                target_label_a = torch.randint_like(real_label_a, 2)
                target_label_b = torch.randint_like(real_label_b, 2)
                mask_a = torch.zeros_like(real_label_a)
                mask_b = torch.zeros_like(real_label_b)
                # domain label
                target_domain_a = torch.cat([target_label_a, mask_b], dim=1)
                target_domain_b = torch.cat([mask_a, target_label_b], dim=1)
                real_domain_a = torch.cat([real_label_a, mask_b], dim=1)
                real_domain_b = torch.cat([mask_a, real_label_b], dim=1)
                # fake from real and target domain
                fake_a = self.gen(real_a, target_domain_a)
                fake_b = self.gen(real_b, target_domain_b)
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
                fake_a = self.gen(real_a, target_domain_a)
                fake_b = self.gen(real_b, target_domain_b)
                # output from disc
                src_fake_a, cls_fake_a = self.disc(fake_a)
                src_fake_b, cls_fake_b = self.disc(fake_b)
                # calc adv loss
                adv_loss = calc_adversal_loss(src_fake_a, 1) + calc_adversal_loss(src_fake_b, 1)
                total_loss[2] += adv_loss.item()
                # apply mask and reshape
                cls_fake_a = cls_fake_a.view(cls_fake_a.shape[:2])
                cls_fake_b = cls_fake_b.view(cls_fake_b.shape[:2])
                cls_fake_a[:,self.len_a:] = torch.zeros_like(cls_fake_a[:,self.len_a:])
                cls_fake_b[:,:self.len_a] = torch.zeros_like(cls_fake_b[:,:self.len_a])
                # calc cls loss
                cls_loss_a = calc_domain_cls_loss(cls_fake_a, target_domain_a)
                cls_loss_b = calc_domain_cls_loss(cls_fake_b, target_domain_b)
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
                gen_loss = adv_loss + self.opt.lambda_cls * cls_loss + self.opt.lambda_rec * rec_loss
                self.optimizer_gen.zero_grad()
                gen_loss.backward()
                self.optimizer_gen.step()

            history['disc_loss'].append(total_loss[0])
            history['gen_loss'].append(total_loss[1])
            print(total_loss)
            # print(f'epoch {epoch} :: disc_loss : {total_loss[0]:.3f}, gen_loss : {total_loss[1]:.3f}')
            save_image(real_a[0], '_real.png')
            save_image(fake_a[0], '_fake.png')
            save_image(rec_a[0], '_rec.png')

            torch.save(self.gen, 'gen.pt')
            torch.save(self.disc, 'disc.pt')

def main():
    opt = utils.parse_opt()

    train = Train(opt)
    train.train()
    

if __name__ == "__main__":
    main()
