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

from dataset import ImageDataset
from handler import ModelHandler

# read config.yaml
config_path = 'config.yaml'


with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
    train_image_path = config['general']['train_image_path']
    val_image_path = config['general']['val_image_path']
    test_image_path = config['general']['test_image_path']
    training_crop_size = config['general']['training_crop_size']
    validation_crop_size = config['general']['validation_crop_size']
    test_crop_size = config['general']['test_crop_size']
    batch_size = config['general']['batch_size']
    resize_mode = config['general']['resize_mode']
    num_epochs = config['general']['num_epochs']
    num_workers = config['general']['num_workers']
    optimizer = config['general']['optimizer']
    learning_rate = config['general']['learning_rate']
    device = config['general']['device']
    seed = config['general']['seed']
    antialias = config['general']['antialias']
    start_channels = config['pretraining']['num_channels']
    end_channels = config['prunning']['num_channels']
    if_pretraining = config['pretraining']['enabled']
    if_prunning = config['prunning']['enabled']
    if_bias_removal = config['bias_removal']['enabled']

train_loader = DataLoader(ImageDataset(train_image_path, training_crop_size, resize_mode=resize_mode, antialias=False, augument=True),
                          batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(ImageDataset(val_image_path, validation_crop_size, resize_mode=resize_mode, antialias=False, cache=True),
                          batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(ImageDataset(test_image_path, test_crop_size, resize_mode=resize_mode, antialias=False),
                          batch_size=batch_size, shuffle=False, num_workers=num_workers)


if if_pretraining:
    handler = ModelHandler(config_path)
    handler.build_models()
    handler.set_stage('pretraining')
    handler.preparation()
    handler.train(train_loader)
    psnr_result = handler.validation(test_loader)
    print(f'PSNR: {psnr_result}')


if if_prunning:
    handler.set_stage('prunning')
    msak0_result = []
    mask1_result = []
    for i in range(start_channels):
        handler.set_mask(0, i)
        msak0_result.append(handler.validation(val_loader))
        handler.remove_mask()
    mask0_result = np.array(msak0_result)
    largest_indices_mask0 = np.argsort(mask0_result)[-2:].tolist()
    handler.channel_removal(0, largest_indices_mask0)
    for i in range(start_channels):
        handler.set_mask(1, i)
        mask1_result.append(handler.validation(val_loader))
        handler.remove_mask()
    mask1_result = np.array(mask1_result)
    largest_indices_mask1 = np.argsort(mask1_result)[-2:].tolist()
    handler.channel_removal(1, largest_indices_mask1)
    print("Indices of largest 2 elements in mask0_result:", largest_indices_mask0)
    print("Indices of largest 2 elements in mask1_result:", largest_indices_mask1)
    handler.train(train_loader)
    psnr_result = handler.validation(test_loader)
    print(f'PSNR: {psnr_result}')


if if_bias_removal:
    handler.set_stage('bias_removal')
    handler.remove_bias()
    handler.train(train_loader)

psnr_result = handler.validation(test_loader)
print(f'finally resulted PSNR: {psnr_result}')
