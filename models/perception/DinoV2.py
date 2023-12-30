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

        self.new_height = (360 // 14) * 14
        self.new_width = (640 // 14) * 14

    def forward(self, x):
        # Original shape: [batch_size, sequence_length, channels, height, width]
        batch_size, seq_len, c, h, w = x.size()

        # Reshape to [batch_size * sequence_length, channels, height, width]
        x = x.view(batch_size * seq_len, c, h, w)

        # Pass through DinoV2 model
        with torch.no_grad():
            outputs = self.model(x)
            
        outputs = outputs.view(batch_size, seq_len, -1)

        return outputs

