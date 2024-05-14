import os
import cv2
import torch
import numpy as np
import torchvision.transforms.functional as FF
import dataset
from PIL import Image

# Define a test directory with sample images
test_dir = './test_imgs'
os.makedirs(test_dir, exist_ok=True)
image1_path = os.path.join(test_dir, 'image1.jpg')
image2_path = os.path.join(test_dir, 'image2.jpg')
cv2.imwrite(image1_path, np.zeros((100, 100, 3), dtype=np.uint8))
cv2.imwrite(image2_path, np.ones((100, 100, 3), dtype=np.uint8) * 255)

# Create an instance of ImageDataset
datasets = dataset.ImageDataset(test_dir, crop_size=64, resize_mode="bilinear", antialias=False)

# Test __len__ method
assert len(datasets) == 2

# Test __getitem__ method
image, image_s = datasets[0]
assert isinstance(image, torch.Tensor)
assert isinstance(image_s, torch.Tensor)
assert image.shape == (3, 64, 64)
assert image_s.shape == (3, 32, 32)
assert torch.all(image >= 0) and torch.all(image <= 1)
assert torch.all(image_s >= 0) and torch.all(image_s <= 1)

# Clean up the test directory
os.remove(image1_path)
os.remove(image2_path)
os.rmdir(test_dir)



datasets = dataset.ImageDataset("./imgs", crop_size=256, resize_mode="bilinear", antialias=False)
image, image_s = datasets[2]

# Convert image and image_s to PIL images
image_pil = FF.to_pil_image(image)
image_s_pil = FF.to_pil_image(image_s)

# Save the images as PNG files
image_pil.save('image.png')
image_s_pil.save('image_s.png')
