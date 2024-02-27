from collections import OrderedDict
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class MSA(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super(type(self), self).__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.q = nn.Linear(in_dim, out_dim)
        self.k = nn.Linear(in_dim, out_dim)
        self.v = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        x = self.norm(x)

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        x = torch.matmul(q, k.transpose(1,2))
        x = torch.softmax(x, dim=2)
        x = torch.dropout(x, 0.5, False)

        x = torch.matmul(x, v)
        return x


class Classifier(nn.Module):
    def __init__(self, in_ch:int, img_size:list[int], n_patch:int, n_msa:int) -> None:
        super(type(self), self).__init__()
        latent_size = 64
        embeded_size = 32
        n_msa = 3


        assert img_size[0] % n_patch == 0 and img_size[1] % n_patch == 0
        patch_h = img_size[0] // n_patch
        patch_w = img_size[1] // n_patch

        self.input = nn.Sequential(
            Rearrange('b c (h p0) (w p1) -> b (p0 p1) (c h w)', p0=n_patch, p1=n_patch),
            nn.LayerNorm(in_ch * patch_h * patch_w),
            nn.Linear(in_ch * patch_h * patch_w, latent_size),
            nn.LayerNorm(latent_size),
        )

        self.embed = nn.Sequential(
            
        )

        
        self.msa = nn.ModuleList([MSA(latent_size, embeded_size) for _ in range(n_msa)])

        self.mhp = nn.Sequential(
            nn.Linear(embeded_size * n_msa, latent_size),
            nn.Linear(latent_size, latent_size),
            nn.Dropout(0.5)
        )

        self.mlp = nn.Sequential(
            nn.LayerNorm(latent_size),
            nn.Linear(latent_size, latent_size),
            nn.Linear(latent_size, latent_size),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(latent_size, latent_size),
            nn.Linear(latent_size, latent_size),
            nn.Dropout(0.5)
        )

        self.output = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(n_patch**2 * latent_size, 1)
        ) 

        
    def forward(self, x):
        x = self.input(x)
        x_skip = self.embed(x)
        x = torch.cat([msa(x_skip) for msa in self.msa], dim=2)
        x_skip = self.mhp(x) + x_skip
        x = self.mlp(x_skip) + x_skip
        x = self.output(x)

        return x