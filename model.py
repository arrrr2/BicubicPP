import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision
import torchvision.transforms as transforms

class ds_sf(nn.Module):
    def __init__(self, ch_in, ch, padding_mode='reflect', bias=True) -> None:
        super().__init__()
        
        self.s2d = nn.PixelUnshuffle(2)
        self.conv = nn.Conv2d((4 * ch_in), ch, 3, 1, 1, padding_mode=padding_mode, bias=bias)

    def forward(self, x):
        x = self.s2d(x)
        x = self.conv(x)
        return x


class wt(nn.Module):
    def __init__(self, ch_in, wt_mode='dwt', fact=2.) -> None:
        super().__init__()
        
        if wt_mode == 'dwt': self.wt = nn.Conv2d(ch_in, ch_in * 4, kernel_size=2, stride=2, bias=False)
        else: self.wt = nn.ConvTranspose2d(ch_in * 4, ch_in, kernel_size=1, stride=2, bias=False)
        
        wt_param = torch.tensor([[[.5, .5], [.5, .5]], [[.5, -.5], [.5, -.5]],
                                  [[.5, .5], [-.5, -.5]], [[.5, -.5], [-.5, .5]]],
                                    dtype=torch.float32)
        
        param_shape = (4 * ch_in, 1 * ch_in, 2, 2)
        param = torch.zeros(param_shape, dtype=torch.float32)
        for i in range(ch_in): param[i*4:(i*4)+4, i, :, :] = wt_param

        if wt_mode == 'dwt': param = param / fact
        else: param = param * fact
        
        self.wt.weight = nn.Parameter(param, requires_grad=False)

    def forward(self, x):
        y = self.wt(x)
        return y
    

class ds_wt(nn.Module):
    def __init__(self, ch_in, ch, padding_mode='reflect', bias=True) -> None:
        super().__init__()
        
        self.dwt = wt(ch_in)
        self.conv = nn.Conv2d((ch_in * 4), ch, 3, 1, 1, padding_mode=padding_mode, bias=bias)

    def forward(self, x):
        x = self.dwt(x)
        x = self.conv(x)
        return x


class ds_conv(nn.Module,):
    def __init__(self, ch_in, ch, padding_mode='reflect', bias=True) -> None:
        super().__init__()
        
        self.conv = nn.Conv2d(ch_in, ch, 3, 2, 1, padding_mode=padding_mode, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        return x


class us(nn.Module):
    def __init__(self, scale) -> None:
        super().__init__()
        self.shuffle = nn.PixelShuffle(scale * 2)

    def forward(self, x):
        return self.shuffle(x)
    
class resblock(nn.Module):
    def __init__(self, ch=32, M=2, relu=nn.LeakyReLU(), padding_mode='reflect', bias=True) -> None:
        super().__init__()
        self.ch = ch
        self.M = M
        self.relu = relu
        self.bias = bias
        self.padding_mode = padding_mode
        self.blocks = self.gen()
        
    def forward(self, x):
        y = self.blocks(x)
        return y + x
    
    def gen(self):
        blocks = []
        for _ in range(self.M):
            blocks.append(nn.Conv2d(self.ch, self.ch, 3, 1, 1, bias=self.bias, padding_mode=self.padding_mode))
            blocks.append(self.relu)
        return nn.Sequential(*blocks)
    

class bicubic_pp(nn.Module):
    def __init__(self, scale=3, R=1, ch=32, M=2, ds="conv", ch_in=3,
                 relu=nn.LeakyReLU(), padding_mode='reflect', bias=True) -> None:
        super().__init__()
        if ds == "conv": self.ds = ds_conv(ch_in, ch, padding_mode, bias)
        elif ds == "dwt": self.ds = ds_wt(ch_in, ch, padding_mode, bias)
        elif ds == "sf": self.ds = ds_sf(ch_in, ch, padding_mode, bias)
        self.mid = nn.Sequential(*[resblock(ch, M, relu, padding_mode, bias) for i in range(R)])
        self.conv = nn.Conv2d(ch, (scale * scale * 4 * ch_in), 3, 1, 1, padding_mode=padding_mode, bias=bias)
        self.up = us(scale)

    def forward(self, x):
        x = self.ds(x)
        y = self.mid(x)
        y = y + x
        y = self.conv(y)
        y = self.up(y)
        return y

class bicubic_pp_prunnable(nn.Module):
    def __init__(self, scale=3, ch=32, ch_in=3,
                 relu=nn.LeakyReLU(), padding_mode='reflect', bias=True):
        super(bicubic_pp_prunnable, self).__init__()
        self.ch_in = ch_in
        self.conv0 = nn.Conv2d(ch_in, ch, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode)
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.conv_out = nn.Conv2d(ch, (2*scale)**2 * ch_in, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.Depth2Space = nn.PixelShuffle(2*scale)
        self.act = relu
        self.padding_mode = padding_mode
        self.mask_layer = None


    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.act(x0)
        masks = [torch.ones((x0.shape[0], x0.shape[1], x0.shape[2], x0.shape[3]), device=x0.device),
                 torch.ones((x0.shape[0], self.conv1.out_channels, x0.shape[2], x0.shape[3]), device=x0.device)]
        if self.mask_layer is not None: masks[self.mask_layer[0]][:, self.mask_layer[1]] = 0.
        x0 = x0 * masks[0]
        x1 = self.conv1(x0)
        x1 = self.act(x1)
        x1 = x1 * masks[1] 
        x2 = self.conv2(x1)
        x2 = self.act(x2) + x0
        x2 = x2 * masks[0]
        y = self.conv_out(x2)
        y = self.Depth2Space(y)
        return y
    
    def set_mask(self, x, y):
        self.mask_layer = (x, y)

    def remove_mask(self):
        self.mask_layer = None
    
    def prune_layers(self, mask_id, prune_channels):

        def prune_certain_layers(conv:nn.Conv2d, prune_channels, prune_type):
            if prune_type == 'input':
                conv.weight = nn.Parameter(torch.cat([conv.weight.data[:, i:i+1] for i in range(conv.in_channels) if i not in prune_channels], dim=1))
                conv.in_channels = conv.in_channels - len(prune_channels)
                print(conv.weight.shape)
            elif prune_type == 'output':
                conv.weight = nn.Parameter(torch.cat([conv.weight.data[i:i+1] for i in range(conv.out_channels) if i not in prune_channels], dim=0))
                if conv.bias is not None:
                    conv.bias = nn.Parameter(torch.cat([conv.bias.data[i:i+1] for i in range(conv.out_channels) if i not in prune_channels], dim=0))
                conv.out_channels = conv.out_channels - len(prune_channels)

        if mask_id == 0:
            prune_certain_layers(self.conv0, prune_channels, 'output')
            prune_certain_layers(self.conv1, prune_channels, 'input')
            prune_certain_layers(self.conv2, prune_channels, 'output')
            prune_certain_layers(self.conv_out, prune_channels, 'input')

        elif mask_id == 1:
            prune_certain_layers(self.conv2, prune_channels, 'input')
            prune_certain_layers(self.conv1, prune_channels, 'output')

    