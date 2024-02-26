from argparse import ArgumentParser, Namespace
from data import CelebaDataLoader, Transformer

def args_parse() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--IMG_HEIGHT", type=int, default=128)
    parser.add_argument("--IMG_WIDTH", type=int, default=128)

    parser.add_argument("--PATH_IMG", type=str, default="CelebA/Img/img_align_celeba")
    parser.add_argument("--PATH_LABEL", type=str, default="CelebA/Anno/list_attr_celeba.txt")
    parser.add_argument("--TRAIN", type=bool, default=True)
    parser.add_argument("--BATCH_SIZE", type=int, default=64)

    return parser.parse_args()


class Trainer:
    def __init__(self, opt:Namespace) -> None:
        self.opt = opt
        self.transformer = Transformer((opt.IMG_HEIGHT, opt.IMG_WIDTH))
        self.dataloader = CelebaDataLoader(opt.PATH_IMG, opt.PATH_LABEL, self.transformer, opt.TRAIN, opt.BATCH_SIZE)

        
        

def main():
    opt = args_parse()
    train = Trainer(opt)


if __name__ == "__main__":
    main()