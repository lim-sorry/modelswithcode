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

    parser.add_argument("--PATH_IMG", type=str, default="/root/CelebA/Img/img_align_celeba")
    parser.add_argument("--PATH_LABEL", type=str, default="/root/CelebA/Anno/list_attr_celeba.txt")
    parser.add_argument("--TRAIN", type=bool, default=True)
    
    parser.add_argument("--EPOCH", type=int, default=128)
    parser.add_argument("--BATCH_SIZE", type=int, default=32)
    parser.add_argument("--LR", type=float, default=0.0001)

    parser.add_argument("--N_PATCH", type=int, default=16)

    return parser.parse_args()


class Trainer:
    def __init__(self, opt:Namespace) -> None:
        opt = opt
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        transformer = Transformer(opt.UP_SIZE, opt.DOWN_SIZE)
        dataloader = CelebaDataLoader(opt.PATH_IMG, opt.PATH_LABEL, transformer, opt.TRAIN, opt.BATCH_SIZE)

        model = SuperSampler(opt.IMG_CH, opt.UP_SIZE, opt.DOWN_SIZE, opt.N_PATCH).to(device)
        optimizer = torch.optim.Adam(model.parameters(), opt.LR)
        criterion = nn.MSELoss()

        ep = 0
        if os.path.exists('supersampler.pt'):
            checkpoint = torch.load('supersampler.pt')
            ep = checkpoint['ep']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])

        for ep in range(ep, opt.EPOCH):
            total_loss = 0.0
            tq = tqdm.tqdm(dataloader, ncols=150)
            for it, (img, img_trg) in enumerate(tq):
                img = img.to(device)
                img_trg = img_trg.to(device)
                img_pred = model(img)

                loss = criterion(img_trg, img_pred)
                total_loss += loss.item()
                tq.set_postfix({'loss':total_loss/(it+1)})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # if it % 100 == 0:
                #     imgs = []
                #     for i in range(4):
                #         imgs.append(resize(img[i], (opt.UP_SIZE, opt.UP_SIZE), InterpolationMode.NEAREST))
                #         imgs.append(resize(img[i], (opt.UP_SIZE, opt.UP_SIZE), InterpolationMode.BICUBIC, antialias=True))
                #         imgs.append(img_pred[i])
                #         imgs.append(img_trg[i])
                #     imgs = make_grid(imgs, 4, 4)
                #     save_image(imgs*0.5+0.5, f'img/{ep}_{it:06d}.png')
                #     save_image(imgs*0.5+0.5, f'sample.png')
            
            checkpoint = {
                'ep': ep+1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(checkpoint, 'supersampler.pt')

            imgs = []
            for i in range(4):
                imgs.append(resize(img[i], (opt.UP_SIZE, opt.UP_SIZE), InterpolationMode.NEAREST))
                imgs.append(resize(img[i], (opt.UP_SIZE, opt.UP_SIZE), InterpolationMode.BICUBIC, antialias=True))
                imgs.append(img_pred[i])
                imgs.append(img_trg[i])
            imgs = make_grid(imgs, 4, 4)
            save_image(imgs*0.5+0.5, f'img/result.png')


def main():
    opt = args_parse()
    train = Trainer(opt)


if __name__ == "__main__":
    main()