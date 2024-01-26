import os

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CarlaDataset(Dataset):
    def __init__(self, img_folder, controls_folder, sequence_length, backbone='cnn', n_outputs=3):
        self.img_folder = img_folder
        self.controls_folder = controls_folder
        self.sequence_length = sequence_length
        self.backbone = backbone
        self.n_outputs = n_outputs

        self.controls = np.load(os.path.join(controls_folder, 'all_controls.npy'))
        self.img_files = [os.path.join(img_folder, f"{i:05d}.png") for i in range(len(self.controls))]

        if self.backbone == 'dinov2':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard ImageNet normalization
            ])
        else:
            self.transform = transforms.ToTensor()

    def __len__(self):
        # Return number of complete sequences
        return len(self.controls) - (self.sequence_length - 1)
    
    def size(self):
        # Return full number of samples
        return len(self.controls)

    def __getitem__(self, idx):
        images = []

        # Get consecutive sequence of controls
        controls = self.controls[idx : idx + self.sequence_length]

        if self.n_outputs == 1:
            controls = controls[:, 0:1]  # Select steering while keeping the dimension
            
        # Get consecutive sequence of images and apply transformations
        for i in range(self.sequence_length):
            img_path = self.img_files[idx + i]
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            images.append(image)

        # Stack images in a sequence
        images = torch.stack(images)

        # print(images.shape, controls.shape)

        return images, controls
