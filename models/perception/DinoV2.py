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
        # X is a batch of videos with shape: [batches, sequences, channels, height, width]
        batch_size, seq_len = x.size(0), x.size(1)
        # Fuse batch and sequence dimensions before passing images through the model
        # This way each sequence of images is also considered a batch as well
        x = x.view(batch_size * seq_len, *x.shape[2:])

        # DinoV2 model inference
        with torch.no_grad():
            outputs = self.model(x)
            
        # Reshape back to batches of videos
        outputs = outputs.view(batch_size, seq_len, -1)

        return outputs