from collections import OrderedDict
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channel=256) -> None:
        super(type(self), self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.InstanceNorm2d(in_channel),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x) + x
    

class Generator(nn.Module):
    def __init__(self, input_channel:int, height:int, width:int) -> None:
        super(type(self), self).__init__()
        self.input_channel  = input_channel
        self.height = height
        self.width = width
        
        layers = OrderedDict()

        # Down-Sampling layers
        channel = 64
        for i in range(3):
            if i == 0:
                layers[f'down_{i}'] = self._make_conv_block(self.input_channel, channel, 7, 1, 3, 'd')
            else:
                layers[f'down_{i}'] = self._make_conv_block(channel, channel*2, 4, 2, 1, 'd')
                channel *= 2
            
        # Residual blocks
        for i in range(6):
            layers[f'res_{i}'] = ResidualBlock()

        # Up-Sampling layers
        for i in range(3):
            if i != 2:
                layers[f'up_{i}'] = self._make_conv_block(channel, channel//2, 4, 2, 1, 'u')
                channel //= 2
            else:
                layers[f'up_{i}'] = self._make_conv_block(channel, 3, 7, 1, 3, 'd', True)
        
        self.model = nn.Sequential(layers)
    
    def forward(self, x):

        return self.model(x)

    def _make_conv_block(self, in_channel:int, out_channel:int, kernel:int, stride:int, padding:int, mode:str, output_layer=False):
        layer = []
        if mode.lower() == 'd':
            layer.append(nn.Conv2d(in_channel, out_channel, kernel, stride, padding))
        elif mode.lower() == 'u':
            layer.append(nn.ConvTranspose2d(in_channel, out_channel, kernel, stride, padding))
        
        if not output_layer:
            layer.append(nn.InstanceNorm2d(out_channel))
            layer.append(nn.ReLU())
        else:
            layer.append(nn.Tanh())

        return nn.Sequential(*layer)





        
    
