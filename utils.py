import argparse


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_celeba', type=str, default='CelebA')
    parser.add_argument('--root_rafd', type=str, default='Rafd')

    parser.add_argument('--height', type=int, default=192)
    parser.add_argument('--width', type=int, default=192)
    parser.add_argument('--channel', type=int, default=3)

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epoch', type=int, default=512)
    

    return parser.parse_args()