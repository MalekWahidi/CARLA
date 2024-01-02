import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class DinoV2(nn.Module):
    def __init__(self, model_name='facebook/dinov2-small'):
        super().__init__()

        # Load DinoV2 Small (with Registers)
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        self.model.eval().cuda()

    def forward(self, x):
        # Input batch shape: [batch_size, sequence_length, channels, height, width]
        b, s, c, h, w = x.size()

        # Reshape to fuse batch and sequence dimensions
        x = x.view(b * s, c, h, w)

        # DinoV2 model inference
        with torch.no_grad():
            outputs = self.model(x)
            
        # Seperate batch and sequence dimensions again
        outputs = outputs.view(b, s, -1)

        return outputs