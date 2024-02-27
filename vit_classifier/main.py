from argparse import ArgumentParser, Namespace
import torch.nn as nn
import torch
import tqdm
from data import CelebaDataLoader, Transformer
from model import Classifier

def args_parse() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--IMG_HEIGHT", type=int, default=128)
    parser.add_argument("--IMG_WIDTH", type=int, default=128)
    parser.add_argument("--IMG_CH", type=int, default=3)

    parser.add_argument("--PATH_IMG", type=str, default="/root/CelebA/Img/img_align_celeba")
    parser.add_argument("--PATH_LABEL", type=str, default="/root/CelebA/Anno/list_attr_celeba.txt")
    parser.add_argument("--TRAIN", type=bool, default=True)
    
    parser.add_argument("--BATCH_SIZE", type=int, default=64)
    parser.add_argument("--LR", type=float, default=0.0001)

    parser.add_argument("--N_PATCH", type=int, default=8)
    parser.add_argument("--N_MSA", type=int, default=3)

    return parser.parse_args()


class Trainer:
    def __init__(self, opt:Namespace) -> None:
        opt = opt
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        img_size = [opt.IMG_HEIGHT, opt.IMG_WIDTH]
        transformer = Transformer(img_size)
        dataloader = CelebaDataLoader(opt.PATH_IMG, opt.PATH_LABEL, transformer, opt.TRAIN, opt.BATCH_SIZE)

        model = Classifier(opt.IMG_CH, img_size, opt.N_PATCH, opt.N_MSA).to(device)
        optimizer = torch.optim.Adam(model.parameters(), opt.LR)
        criterion = nn.MSELoss()

        for _ in range(128):
            total_loss = 0.0
            tq = tqdm.tqdm(dataloader)
            for i, (img, label) in enumerate(tq):
                img = img.to(device)
                label = label.to(device)
                output = model(img)
                loss = criterion(label, output)
                total_loss += loss.item()
                tq.set_postfix({'loss':total_loss/(i+1)})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(total_loss)
        

def main():
    opt = args_parse()
    train = Trainer(opt)


if __name__ == "__main__":
    main()