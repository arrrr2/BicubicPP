import torch
# a regular training pipeline

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as FF
import os
import numpy as np
import cv2
import yaml
from model import bicubic_pp
import time
from torch.quantization import quantize_dynamic, quantize, fuse_modules, quantize_dynamic_jit

def get_fp16(in_data: torch.Tensor|nn.Module):
    return in_data.half()

def make_pair(r=1, m=2, ch=32, bias=False, cuda=False, fp16=False):
    model = bicubic_pp(R=r, M=m, ch = ch, bias=bias, padding_mode='zeros').cpu().float()
    input = torch.rand((1, 3, 720, 1280)).cpu().float()
    if cuda: model, input = model.cuda(), input.cuda()
    # else:
        # model = quantize_dynamic(model,{torch.nn.Conv2d}, dtype=torch.qint8, inplace=True)
        # model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        # model = torch.quantization.prepare(model)
        # model = torch.quantization.convert(model)

        
    if fp16: model, input = model.half(), input.half()
    return model, input
    
def test_time(ch, bias, cuda, fp16, r=1, m=1):
    model, input = make_pair(r, m, ch, bias, cuda, fp16, )
    for rg in range(80): _ = model(input); torch.cuda.synchronize()
    torch.cuda.synchronize()
    t = time.time()
    for rg in range(1000): _ = model(input); torch.cuda.synchronize()
    torch.cuda.synchronize()
    return time.time() - t


@torch.no_grad()
def test_speed(t):
    for ch in range(32, 33):
        t_s = []
        for bias in [False]:
            for cuda in [True]:
                for fp16 in [False]:
                    t_s.append(test_time(ch, bias, cuda, fp16))
        if t != 0: print(ch, *[f"{t/2:.3f}" for t in t_s], sep=',')
 
 
@torch.no_grad()  
def test_r_m():
    for r in range(1, 5):
        t_s = []
        for m in range(1, 5):
            t_s.append(test_time(32, False, False, False, r=r, m=m))
        print(r, *[f"{t:.3f}" for t in t_s], sep=',')
            
   
# torch.set_num_threads(4)     
# time.sleep(3)
# for t in range(0,2): test_speed(t)
test_r_m()
print('\n\n\n')
test_r_m()