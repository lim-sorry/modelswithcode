from argparse import ArgumentParser, Namespace
import os
import torch.nn as nn
import torch
import tqdm
from data import CelebaDataLoader, Transformer
from model import SuperSampler
from torchvision.utils import save_image, make_grid
from torchvision.transforms.functional import resize, InterpolationMode

def args_parse() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--UP_SIZE", type=int, default=192)
    parser.add_argument("--DOWN_SIZE", type=int, default=48)
    parser.add_argument("--IMG_CH", type=int, default=3)

    parser.add_argument("--TRAIN", type=bool, default=True)
    parser.add_argument("--PATH_PT", type=str, default="supersampler_01.pt")
    parser.add_argument("--PATH_IMG", type=str, default="/root/CelebA/Img/img_align_celeba")
    parser.add_argument("--PATH_LABEL", type=str, default="/root/CelebA/Anno/list_attr_celeba.txt")
    
    parser.add_argument("--EPOCH", type=int, default=1)
    parser.add_argument("--BATCH_SIZE", type=int, default=32)
    parser.add_argument("--LR", type=float, default=0.0001)

    parser.add_argument("--N_PATCH", type=int, default=16)

    return parser.parse_args()


class Trainer:
    def __init__(self, opt:Namespace) -> None:
        self.opt = opt
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.img_size = [self.opt.UP_SIZE]*2
        
        self.transformer = Transformer(opt.UP_SIZE, opt.DOWN_SIZE)
        self.dataloader = CelebaDataLoader(opt.PATH_IMG, opt.PATH_LABEL, self.transformer, opt.TRAIN, opt.BATCH_SIZE)
        
        self.ep = 0
        self.model = SuperSampler(opt.IMG_CH, opt.UP_SIZE, opt.DOWN_SIZE, opt.N_PATCH).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), opt.LR)
        self.criterion = nn.MSELoss()

        if os.path.exists(opt.PATH_PT):
            checkpoint = torch.load(opt.PATH_PT)
            self.ep = checkpoint['ep']
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def train(self):
        self.model.train()
        ep = self.ep
        for ep in range(ep, self.opt.EPOCH):
            total_loss = 0.0
            tq = tqdm.tqdm(self.dataloader, ncols=150)
            for it, (img, img_trg) in enumerate(tq):
                img = img.to(self.device)
                img_trg = img_trg.to(self.device)
                img_pred = self.model(img)

                loss = self.criterion(img_trg, img_pred)
                total_loss += loss.item()
                tq.set_postfix({'loss':total_loss/(it+1)})

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
                if it%100 == 0 or it==len(self.dataloader)-1:
                    imgs = []
                    for i in range(4):
                        imgs.append(resize(img[i], self.img_size, InterpolationMode.NEAREST))
                        imgs.append(resize(img[i], self.img_size, InterpolationMode.BICUBIC, antialias=True))
                        imgs.append(img_pred[i])
                        imgs.append(img_trg[i])
                    imgs = make_grid(imgs, 4, 4)
                    save_image(imgs*0.5+0.5, f'img/result.png')

            checkpoint = {
                'ep': ep+1,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }
            torch.save(checkpoint, self.opt.PATH_PT)

    def test(self):
        with torch.no_grad():
            total_loss = 0.0        
            tq = tqdm.tqdm(self.dataloader, ncols=150)
            for img, img_trg in tq:
                img = img.to(self.device)
                img_trg = img_trg.to(self.device)
                img_pred = self.model(img)
                loss = self.criterion(img_trg, img_pred)
                total_loss += loss
                tq.set_postfix({'loss':total_loss.item()})

        imgs = []
        for i in range(4):
            imgs.append(resize(img[i], self.img_size, InterpolationMode.NEAREST))
            imgs.append(resize(img[i], self.img_size, InterpolationMode.BICUBIC, antialias=True))
            imgs.append(img_pred[i])
            imgs.append(img_trg[i])
        imgs = make_grid(imgs, 4, 4)
        save_image(imgs*0.5+0.5, f'img/test_result.png')

def main():
    opt = args_parse()

    trainer = Trainer(opt)
    if opt.TRAIN == True:
        trainer.train()
    else:
        trainer.test()


if __name__ == "__main__":
    main()