import argparse


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_celeba', type=str, default='CelebA')
    parser.add_argument('--root_rafd', type=str, default='Rafd')
    
    parser.add_argument('--name_a', type=str, default='celeba')
    parser.add_argument('--name_b', type=str, default='celeba')
    parser.add_argument('--label_a', type=list, default=['Black_Hair', 'Blond_Hair', 'Brown_Hair'])
    parser.add_argument('--label_b', type=list, default=['Young', 'Male'])

    parser.add_argument('--lambda_cls', type=float, default=1.0)
    parser.add_argument('--lambda_rec', type=float, default=10.0)

    parser.add_argument('--height', type=int, default=128)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--channel', type=int, default=3)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=1024)
    
    return parser.parse_args()
