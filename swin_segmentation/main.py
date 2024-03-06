import os
import torch
import torch.nn as nn
from argparse import ArgumentParser, Namespace

import tqdm
from function import Transform
from model import SwinModel
from torchvision.utils import save_image, make_grid
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader

def parse() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')

    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--n_cat', type=int, default=2)

    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--n_ch', type=int, default=96)
    parser.add_argument('--w_size', type=int, default=7)
    parser.add_argument('--n_head', type=int, default=3)

    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-3)

    parser.add_argument('--epoch', type=int, default=1024)
    parser.add_argument('--pt', type=str, default='segmentation.pt')

    return parser.parse_args()


class Trainer:
    def __init__(self, opt:Namespace) -> None:
        assert (opt.img_size / opt.patch_size) % 8 == 0
        self.opt = opt
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        d_patch = opt.img_ch * opt.patch_size * opt.patch_size

        self.ep = 0
        self.model = SwinModel(d_patch, opt.n_ch, opt.w_size, opt.n_head, opt.n_cat).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr, weight_decay=opt.wd)      

        if os.path.exists(opt.pt):
            checkpoint = torch.load(opt.pt)
            self.ep = checkpoint['ep']
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.transform = Transform(opt.img_size)
        self.criterion = nn.MSELoss()
        # self.criterion = nn.BCELoss()

    def train(self):
        opt = self.opt

        dataset = OxfordIIITPet(opt.data_path, 'trainval', 'segmentation', self.transform, download=True)
        dataloader = DataLoader(dataset, opt.batch_size, shuffle=True, drop_last=True)

        for ep in range(self.ep, opt.epoch):
            total_loss = 0.0
            tq = tqdm.tqdm(dataloader)
            for i, (img, trg) in enumerate(tq):
                img = img.to(self.device)
                trg = trg.to(self.device)
                out = self.model(img)
                loss = self.criterion(out, trg)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                tq.set_postfix_str(f'total loss : {total_loss/(i+1):6.3f}')

                if i%100==0:
                    img_trg = img[:8]*0.5+0.5+out[:8]
                    img_pred = img[:8]*0.5+0.5+trg[:8]
                    save_image(make_grid(torch.cat([img_trg, img_pred], 0), nrow=4), 'segmentation.png')
            
            checkpoint = {
                'ep': ep+1,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
            torch.save(checkpoint, opt.pt)

    def test(self):
        opt = self.opt

        dataset = OxfordIIITPet(opt.data_path, 'test', 'segmentation', self.transform, download=True)
        dataloader = DataLoader(dataset, opt.batch_size, shuffle=False, drop_last=True)
        for img, trg in dataloader:
            img = img.to(self.device)
            trg = trg.to(self.device)
            out = self.model(img)
            
            img = img+out
            save_image(make_grid(img[:16], nrow=4), 'segmentation.png')
            break


def main() -> None:
    opt = parse()
    
    trainer = Trainer(opt)
    if opt.mode.lower() == 'train':
        trainer.train()
    if opt.mode.lower() == 'test':
        trainer.test()



if __name__ == '__main__':
    main()