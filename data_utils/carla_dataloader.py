import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CarlaDataset(Dataset):
    def __init__(self, img_folder, controls_folder, sequence_length):
        self.img_folder = img_folder
        self.controls_folder = controls_folder
        self.sequence_length = sequence_length

        self.controls = np.load(os.path.join(controls_folder, 'all_controls.npy'))
        self.img_files = [os.path.join(img_folder, f"{i:05d}.png") for i in range(len(self.controls))]

    def __len__(self):
        # Adjust to ensure only complete sequences are included
        return len(self.controls) - (self.sequence_length - 1)

    def __getitem__(self, idx):
        images = []
        controls = self.controls[idx:idx + self.sequence_length]

        for i in range(self.sequence_length):
            img_path = self.img_files[idx + i]
            image = Image.open(img_path).convert('RGB')
            image = transforms.ToTensor()(image)
            images.append(image)

        images = torch.stack(images)

        return images, controls