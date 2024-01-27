import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ConditionalCarlaData(Dataset):
    def __init__(self, hdf5_file, sequence_length, num_commands=4, backbone='cnn'):
        self.hdf5_file = hdf5_file
        # print("hdf5_file: ", hdf5_file)
        self.sequence_length = sequence_length
        self.num_commands = num_commands
        self.backbone = backbone
        self.transform = self._create_transform(backbone)

    def _create_transform(self, backbone):
        # Transformations
        if backbone == 'dinov2':
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            return transforms.ToTensor()

    def __len__(self):
        # Return number of complete sequences
        with h5py.File(self.hdf5_file, 'r') as h5_file:
            return len(h5_file['targets']) - (self.sequence_length - 1)

    def size(self):
        print("measuring size")
        # Return full number of samples
        with h5py.File(self.hdf5_file, 'r') as h5_file:
            return len(h5_file['targets'][0])

    def close(self):
        self.h5_file.close()
    
    def __getitem__(self, idx):
        images = []
        controls = np.zeros((self.sequence_length, 3), dtype=np.float32)  # Assuming control data has 3 values per sample
        commands = np.zeros((self.sequence_length, self.num_commands), dtype=np.float32)  # Assuming commands are one-hot encoded

        with h5py.File(self.hdf5_file, 'r') as h5_file:
            for i in range(self.sequence_length):
                # Read image directly from HDF5 file
                image_data = h5_file['rgb'][idx + i]
                image = Image.fromarray(image_data).convert('RGB')
                image = self.transform(image)
                images.append(image)

                # Extract controls (steering, gas, brake)
                controls[i, :] = h5_file['targets'][idx + i, :3]

                # One-hot encode the high-level command
                # Follow lane (2): [1, 0, 0, 0]
                # Left (3): [0, 1, 0, 0]
                # Right (4): [0, 0, 1, 0]
                # Straight (5): [0, 0, 0, 1]
                command_idx = int(h5_file['targets'][idx, -4]) - 2
                commands[i, command_idx] = 1

        # Convert sequences to pytorch tensors
        images = torch.stack(images)
        controls = torch.tensor(controls, dtype=torch.float32)
        commands = torch.tensor(commands, dtype=torch.float32)

        # Return sequence of images, controls, and high-level commands
        return images, commands, controls
