import torch
import torch.nn as nn

from PIL import Image
from utils import vc1_utils

class VC1(nn.Module):
    def __init__(self, model_name='facebook/dinov2-small'):
        super().__init__()

        # Load VC-1 (base)
        self.model, self.embd_size, self.preprocessing, self.model_info = vc1_utils.load_model(vc1_utils.VC1_BASE_NAME)
        self.model.eval().cuda()

    def forward(self, x):
        # Input batch shape: [batch_size, sequence_length, channels, height, width]
        b, s, c, h, w = x.size()

        # Reshape to fuse batch and sequence dimensions
        x = x.view(b * s, c, h, w)

        x = self.preprocessing(x) # Outputs will be (B, 3, 224, 224)

        # VC-1 model inference
        with torch.no_grad():
            outputs = self.model(x) # outputs is 1x768
        
        # Seperate batch and sequence dimensions again
        outputs = outputs.view(b, s, -1)

        return outputs