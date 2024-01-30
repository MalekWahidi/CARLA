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
        # X is a batch of videos with shape: [batches, sequences, channels, height, width]
        batch_size, seq_len = x.size(0), x.size(1)
        # Fuse batch and sequence dimensions before passing images through the model
        # This way each sequence of images is also considered a batch as well
        x = x.view(batch_size * seq_len, *x.shape[2:])

        # Preprocessed input should be (B, 3, 224, 224)
        x = self.preprocessing(x) 

        # VC-1 model inference
        with torch.no_grad():
            outputs = self.model(x) # outputs is 1x768
        
        # Reshape back to batches of videos
        outputs = outputs.view(batch_size, seq_len, -1)

        return outputs