import os
import torch
from argparse import ArgumentParser, Namespace
from function import Transform
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader

def parse() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')

    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--data_path', type=str, default='~/data')
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--epoch', type=int, default=1024)
    parser.add_argument('--pt', type=str, default='segmentation.pt')

    return parser.parse_args()


class Trainer:
    def __init__(self, opt:Namespace) -> None:
        self.opt = opt
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.transform = Transform(opt.img_size)


    def train(self):
        opt = self.opt

        dataset = OxfordIIITPet(opt.data_path, 'trainval', 'segmentation', self.transform)
        dataloader = DataLoader(dataset, opt.batch_size, shuffle=True, drop_last=True)

        if os.path.exists(opt.pt):
            checkpoint = torch.load(opt.pt)
            ep = checkpoint['ep']

        
    def test(self):
        pass


def main() -> None:
    opt = parse()
    
    trainer = Trainer(opt)
    if opt.mode.lower() == 'train':
        trainer.train()
    if opt.mode.lower() == 'test':
        trainer.test()



if __name__ == '__main__':
    main()