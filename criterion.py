import torch
import torch.nn as nn


def calc_adversal_loss(src:torch.Tensor, trg:int) -> torch.Tensor:
    src = torch.mean(src.view(src.size(0), -1), 1)
    trg = torch.ones_like(src) if trg==1 else torch.zeros_like(src)
    return nn.functional.mse_loss(src, trg)


def calc_domain_cls_loss(cls:torch.Tensor, trg:torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(cls, trg)


def calc_reconstruction_loss(real:torch.Tensor, rec:torch.Tensor) -> torch.Tensor:
    return nn.functional.l1_loss(real, rec)

