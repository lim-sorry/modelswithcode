import torch
import utils
from torch.utils.data import DataLoader
from dataset import CustomDataset, CustomTransform
from model import Discriminator, Generator
import tqdm
import time

def train(opt) -> None:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    transform = CustomTransform(opt.height, opt.width)
    customDataset = CustomDataset(opt.root_celeba, opt.root_rafd, 'train', transform)
    dataloader = DataLoader(customDataset, opt.batch_size, shuffle=True, drop_last=True, num_workers=4)

    gen = Generator(opt.channel, opt.height, opt.width).to(device)
    print(torch.compile(gen))
    disc = Discriminator(opt.channel, opt.height, opt.width, opt.n_domain).to(device)
    print(torch.compile(disc))
    
    tq = tqdm.tqdm(range(opt.epoch))
    for epoch in tq:
        for batch in dataloader:
            image = batch[0]['image'].to(device)
            output = gen(image)
            output = disc(image)
            break
    print(output[0].shape, output[1].shape)

def main():
    opt = utils.parse_opt()

    train(opt)

if __name__ == "__main__":
    main()

