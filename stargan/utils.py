import argparse

import re
from typing import Dict
import torch
from torchvision.utils import save_image, make_grid

def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_celeba', type=str, default='CelebA')
    parser.add_argument('--root_rafd', type=str, default='Rafd')

    parser.add_argument('--root_path', type=str, default='')
    
    parser.add_argument('--name_a', type=str, default='celeba')
    parser.add_argument('--name_b', type=str, default='celeba')

    # need to modify train.py target label create to change labels
    parser.add_argument('--label_a', type=list, default=['Black_Hair', 'Blond_Hair', 'Brown_Hair'])
    parser.add_argument('--label_b', type=list, default=['Young', 'Male'])

    parser.add_argument('--lambda_cls', type=float, default=1.0)
    parser.add_argument('--lambda_rec', type=float, default=10.0)

    parser.add_argument('--height', type=int, default=128)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--channel', type=int, default=3)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=100)
    
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.00001)

    parser.add_argument('--iter', type=int, default=0)

    return parser.parse_args()


def save_model(gen:torch.nn.Module, disc:torch.nn.Module):
    torch.save(gen, 'gen.pt')
    torch.save(disc, 'disc.pt')

def save_image_grid(image:torch.Tensor, path:str, nrow:int):
    save_image(make_grid(image * 0.5 + 0.5, nrow), path)
