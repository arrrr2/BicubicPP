# a regular dataset class for the dataset, with importing common libraries.
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import torch.nn.functional as FF



class ImageDataset(Dataset):
    def __init__(self, data_dir, crop_size, scale, resize_mode="bilinear", antialias=False, cache=False, augument=False, device='cpu'):
        self.data_dir = data_dir
        self.image_files = [file for file in os.listdir(data_dir) if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        self.crop_size = crop_size
        self.resize_mode = resize_mode
        self.antialias = antialias
        self.if_cache = cache
        self.cache = {}
        self.scale = scale
        self.augument = augument
        self.device = device

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if self.if_cache and self.cache.get(idx) is not None:
            return self.cache[idx]
        image_path = os.path.join(self.data_dir, self.image_files[idx])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        if self.crop_size > 0:
            top = np.random.randint(0, h - self.crop_size)
            left = np.random.randint(0, w - self.crop_size)
            image = image[top:top+self.crop_size, left:left+self.crop_size]

        if self.augument:
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=0)
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=1)
            if np.random.rand() > 0.5:
                image = np.rot90(image)
                
        image = torch.from_numpy(image.copy())    
        img = image.permute(2, 0, 1).to(torch.float32).contiguous()

        image_s = FF.interpolate(img.unsqueeze(0), 
                                 scale_factor=(1/self.scale), mode=self.resize_mode, antialias=self.antialias)

        image = img.to(torch.float32) / 255.
        image_s = image_s.to(torch.float32) / 255.

        image = image.squeeze()
        image_s = image_s.squeeze()

        image = image.contiguous()
        image_s = image_s.contiguous()
        if self.if_cache:
            self.cache[idx] = (image_s, image)
        return image_s, image
