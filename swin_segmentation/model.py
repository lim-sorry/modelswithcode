import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class MSA(nn.Module):
    def __init__(self, n_ch:int):
        super(type(self), self).__init__()
        self.sqrt_n_ch = n_ch ** 0.5
        self.layerNorm = nn.LayerNorm(n_ch)
        self.qkv = nn.Linear(n_ch, n_ch*3)

    def forward(self, x):
        x = self.layerNorm(x)

        qkv = self.qkv(x)
        qkv = qkv.unflatten(-1, (-1, 3))
        qkv = qkv.permute(-1,0,1,2,3)
        
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        x = torch.matmul(q, k.transpose(2,3)) / self.sqrt_n_ch
        x += pos_embed.to(x.device)
        x = torch.matmul(x, v)
        
        return x


class TransformerBlock(nn.Module):
    def __init__(self, shift:bool, n_patch:int, n_head:int, n_ch:int, w_size:int):
        super(type(self), self).__init__()
        assert n_patch % w_size == 0
        n_window = n_patch // w_size
        self.shift = shift
        self.w_size = w_size
        self.n_head = n_head
        
        self.part = Rearrange('b (h p0) (w p1) c -> b (h w) (p0 p1) c', p0=w_size, p1=w_size)
        self.msa = nn.ModuleList([
            MSA(n_ch) for _ in range(n_head)
        ])
        self.mhp = nn.Sequential(
            nn.Linear(n_ch*n_head, n_ch)
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(n_ch),
            nn.Linear(n_ch, n_ch*4),
            nn.GELU(),
            nn.Linear(n_ch*4, n_ch),
        )
        self.unpart = Rearrange('b (hw ww) (h w) c -> b (hw h) (ww w) c', ww=n_window, w=w_size)

    
    def forward(self, x):
        if self.shift:
            x = torch.roll(x, (self.w_size//2, self.w_size//2), (1, 2))
        x = self.part(x)
        x_skip = x

        x = torch.cat([self.msa[i](x) for i in range(self.n_head)], dim=-1)
        x = self.mhp(x)
        x = x + x_skip

        x_skip = x
        x = self.mlp(x)
        x = x + x_skip

        x = self.unpart(x)
        if self.shift:
            x = torch.roll(x, (-self.w_size//2, -self.w_size//2), (1, 2))
        
        return x


class SwinModel(nn.Module):
    def __init__(self, d_patch:int, n_ch:int, w_size:int, n_head:int, n_cat:int):
        super(type(self), self).__init__()
        self.d_patch = d_patch
        self.w_size = w_size
        self._set_pos_embed()

        self.input = Rearrange('b c (h p0) (w p1) -> b h w (p0 p1 c)', p0=4, p1=4)

        self.stage_1 = self._make_stage_block('embed', 2, 56, n_head, n_ch)
        self.stage_2 = self._make_stage_block('merge', 2, 56, n_head, n_ch)
        self.stage_3 = self._make_stage_block('merge', 6, 28, n_head, n_ch*2)
        self.stage_4 = self._make_stage_block('merge', 2, 14, n_head, n_ch*4)

        self.up_0 = self._make_up_block(n_ch*8)
        self.up_1 = self._make_up_block(n_ch*8)
        self.up_2 = self._make_up_block(n_ch*6)

        self.up_3 = nn.Sequential(
            nn.ConvTranspose2d(n_ch*4,n_ch*2,4,2,1,0, bias=False),
            nn.GroupNorm(4,n_ch*2),
            nn.GELU(),
            # nn.Conv2d(n_ch*2,n_ch*2,3,1,1,bias=False),
            # nn.GroupNorm(4,n_ch*2),
            # nn.GELU(),
            nn.ConvTranspose2d(n_ch*2,n_ch,4,2,1,0, bias=False),
            nn.GroupNorm(4,n_ch),
            nn.GELU(),
        )
        self.output = nn.Sequential(
            nn.Conv2d(n_ch+3,n_ch,3,1,1,bias=False),
            nn.GroupNorm(4,n_ch),
            nn.GELU(),
            nn.Conv2d(n_ch,3,3,1,1,bias=False),
            nn.Softmax(1)
        )

    def forward(self, x):
        x0 = x
        x = self.input(x)
        x1 = self.stage_1(x)
        x2 = self.stage_2(x1)
        x3 = self.stage_3(x2)
        x = self.stage_4(x3)
        x = x.permute(0,3,1,2)
        x = self.up_0(x)
        x = torch.cat([x, x3.permute(0,3,1,2)], dim=1)
        x = self.up_1(x)
        x = torch.cat([x, x2.permute(0,3,1,2)], dim=1)
        x = self.up_2(x)
        x = torch.cat([x, x1.permute(0,3,1,2)], dim=1)
        x = self.up_3(x)
        x = torch.cat([x, x0], 1)
        x = self.output(x)

        return x
    
    def _make_stage_block(self, mode:str, n_layer:int, n_patch:int, n_head:int, n_ch:int):
        assert mode.lower() in ('merge', 'embed')
        layers = []

        if mode == 'embed':
            layers.append(nn.Linear(self.d_patch, n_ch))
        else:
            n_ch *= 2
            n_patch //= 2
            layers.append(Rearrange('b (h p0) (w p1) c -> b h w (c p0 p1)', p0=2, p1=2))
            layers.append(nn.Linear(n_ch*2, n_ch))
        
        for i in range(n_layer):
            shift = False if i % 2 == 0 else True
            layers.append(TransformerBlock(shift, n_patch, n_head, n_ch, self.w_size))

        return nn.Sequential(*layers)
    

    def _make_up_block(self, n_ch:int):
        return nn.Sequential(
            nn.ConvTranspose2d(n_ch,n_ch//2,4,2,1,0,bias=False),
            nn.GroupNorm(4, n_ch//2),
            nn.GELU(),
            # nn.Conv2d(n_ch//2,n_ch//2,3,1,1,bias=False),
            # nn.GroupNorm(4, n_ch//2),
            # nn.GELU(),
        )


    def _set_pos_embed(self):
        global pos_embed

        axis_x = [[[[i for i in range(k, k+self.w_size)] for _ in range(self.w_size)] for k in range(0,-self.w_size,-1)] for _ in range(self.w_size)]
        axis_x = torch.tensor(axis_x, dtype=float).reshape(self.w_size**2, self.w_size**2)
        
        axis_y = [[[[i - k for _ in range(self.w_size)] for i in range(self.w_size)] for _ in range(self.w_size)] for k in range(self.w_size)]
        axis_y = torch.tensor(axis_y, dtype=float).reshape(self.w_size**2, self.w_size**2)

        axis_x += self.w_size - 1
        axis_y += self.w_size - 1
        axis_y *= 2 * self.w_size - 1

        pos_embed = axis_x + axis_y

if __name__ == '__main__':
    pass