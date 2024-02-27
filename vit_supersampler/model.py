import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import torch.nn.functional as F

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


class PosEmbedBlock(nn.Module):
    def __init__(self, in_ch:int, in_dim:int) -> None:
        super(type(self), self).__init__()
        assert in_dim % 2 == 0

        k = torch.arange(1, (in_dim//2)+1, 1)
        w = torch.float_power(10000, -(2*k/in_dim)).view(-1, in_dim//2).repeat(in_ch, 1)
        t = torch.arange(1, in_ch+1, 1).view(-1,1).repeat_interleave(in_dim//2, 1)
        w_t = w * t
        p_t = torch.stack([torch.sin(w_t), torch.cos(w_t)], dim=1)
        self.p_t = einops.rearrange(p_t, 'c p0 d -> 1 c (d p0)', p0=2).float()
        # print(torch.sin(w_t)[:3][0:3], torch.cos(w_t)[:3][0:3])
        # print(self.p_t[:3][:6])
        
        self.block = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Linear(in_dim, in_dim),
            nn.Dropout(0.5, True)
        )

    def forward(self, x):
        if self.p_t.device != x.device:
            self.p_t = self.p_t.to(x.device)
        x = x + self.p_t
        x = self.block(x)

        return x

class SuperSampler(nn.Module):
    def __init__(self, in_ch:int, up_size:int, down_size:int, n_patch:int) -> None:
        super(type(self), self).__init__()
        self.latent_size = up_size // 4
        latent_dim = self.latent_size**2
        attetion_dim = 32
        n_msa = 16

        assert down_size % n_patch == 0
        patch_size = down_size // n_patch

        self.input = nn.Sequential(
            Rearrange('b c (h p0) (w p1) -> b (p0 p1) (c h w)', p0=n_patch, p1=n_patch),
            nn.LayerNorm(in_ch * patch_size**2),
            nn.Linear(in_ch * patch_size**2, latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # self.embed = PosEmbedBlock(n_patch**2, latent_dim)
        
        self.msa = nn.ModuleList([MSA(latent_dim, attetion_dim) for _ in range(n_msa)])

        self.mhp = nn.Sequential(
            nn.Linear(attetion_dim * n_msa, attetion_dim),
            nn.Linear(attetion_dim, latent_dim),
            nn.Dropout(0.5, True)
        )

        self.mlp = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(0.5, True),
            nn.Linear(latent_dim, latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.Dropout(0.5, True),
        )

        ch = n_patch**2
        self.up_0 = nn.Sequential(
            nn.ConvTranspose2d(ch + in_ch, ch//2, 4, 2, 1, 0),
            nn.GroupNorm(4, ch//2),
            nn.GELU(),
            nn.Conv2d(ch//2, ch//2, 3, 1, 1),
            nn.GroupNorm(4, ch//2),
            nn.GELU(),
        )

        self.up_1 = nn.Sequential(
            nn.ConvTranspose2d(ch//2 + in_ch, ch//4, 4, 2, 1, 0),
            nn.GroupNorm(4, ch//4),
            nn.GELU(),
            nn.Conv2d(ch//4, ch//4, 3, 1, 1),
            nn.GroupNorm(4, ch//4),
            nn.GELU(),
        )

        self.output = nn.Sequential(
            nn.Conv2d(ch//4 + in_ch, in_ch, 1, 1, 0),
            nn.Tanh()
        )

        
    def forward(self, x):
        x_0 = x

        x = self.input(x)
        # x_embed = self.embed(x)
        x_embed = x

        x = torch.cat([msa(x_embed) for msa in self.msa], dim=2)
        x_skip = self.mhp(x) + x_embed
        x = self.mlp(x_skip) + x_skip
        x = x.view(x.size(0), x.size(1), self.latent_size, self.latent_size)

        x = torch.cat([x_0, x], dim=1)
        x = self.up_0(x)

        x_0 = F.interpolate(x_0, scale_factor=2, mode='nearest')
        x = torch.cat([x_0, x], dim=1)
        x = self.up_1(x)
        
        x_0 = F.interpolate(x_0, scale_factor=2, mode='nearest')
        x = torch.cat([x_0, x], dim=1)
        x = self.output(x)

        return x