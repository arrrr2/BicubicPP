# a regular dataset class for the dataset, with importing common libraries.
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import torch.nn.functional as FF



class ImageDataset(Dataset):
    def __init__(self, data_dir, crop_size, resize_mode="bilinear", antialias=False):
        self.data_dir = data_dir
        self.image_files = [file for file in os.listdir(data_dir) if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        self.crop_size = crop_size
        self.resize_mode = resize_mode
        self.antialias = antialias

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_files[idx])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        if self.crop_size > 0:
            top = np.random.randint(0, h - self.crop_size)
            left = np.random.randint(0, w - self.crop_size)
            image = image[top:top+self.crop_size, left:left+self.crop_size]

        image_s = FF.interpolate(torch.from_numpy(image).to(torch.float32).permute(2, 0, 1).unsqueeze(0), 
                                 scale_factor=0.5,mode=self.resize_mode, antialias=self.antialias)

        image = torch.from_numpy(image).permute(2, 0, 1).to(torch.float32) / 255.
        image_s = image_s.to(torch.float32) / 255.

        image = image.squeeze()
        image_s = image_s.squeeze()

        image = image.contiguous()
        image_s = image_s.contiguous()
        
        return image, image_s
    
class ValImageDataset(Dataset):
    def __init__(self, data_dir, crop_size, resize_mode="bilinear", antialias=False):
        self.data_dir = data_dir
        self.image_files = [file for file in os.listdir(data_dir) if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        self.crop_size = crop_size
        self.resize_mode = resize_mode
        self.antialias = antialias
        self.cached_data = []

        for idx in range(len(self.image_files)):
            image_path = os.path.join(self.data_dir, self.image_files[idx])
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, _ = image.shape
            top = np.random.randint(0, h - self.crop_size)
            left = np.random.randint(0, w - self.crop_size)
            image = image[top:top+self.crop_size, left:left+self.crop_size]

            image_s = FF.interpolate(torch.from_numpy(image).to(torch.float32).permute(2, 0, 1).unsqueeze(0), 
                                     scale_factor=0.5,mode=self.resize_mode, antialias=self.antialias)

            image = torch.from_numpy(image).to(torch.float32) / 255.
            image_s = image_s.to(torch.float32) / 255.
            image = image.contiguous()
            image_s = image_s.contiguous()

            self.cached_data.append((image, image_s))

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, idx):
        return self.cached_data[idx]
    
