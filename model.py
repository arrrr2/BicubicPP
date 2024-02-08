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
        y = self.ds(x)
        y = self.mid(y)
        y = self.conv(y)
        y = self.up(y)
        return y

net = bicubic_pp(ds='sf').cpu()
img = torch.zeros((1, 3, 128, 128)).cpu()
res = net(img)
print(res.shape)