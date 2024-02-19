import torch
import utils
from torch.utils.data import DataLoader
from dataset import CustomDataset, CustomTransform

def train(opt) -> None:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    transform = CustomTransform(opt.height, opt.width)
    customDataset = CustomDataset(opt.root_celeba, opt.root_rafd, 'train', transform)
    dataloader = DataLoader(customDataset, opt.batch_size, shuffle=True, drop_last=True)


    


def main():
    opt = utils.parse_opt()

    train(opt)

if __name__ == "__main__":
    main()

