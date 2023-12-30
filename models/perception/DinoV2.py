import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

class DinoV2(nn.Module):
    def __init__(self, model_name='facebook/dinov2-small'):
        super().__init__()
        # Load input processing pipeline
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        
        # Load DinoV2 Small (with Registers)
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        self.model.eval()

    def forward(self, x):
        # Reshape assuming x is a batch of sequences of images
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.view(batch_size * seq_len, *x.shape[2:])

        # Process images and pass through the model
        inputs = self.processor(images=x, return_tensors="pt")
        outputs = self.model(**inputs)

        # Extract the last hidden states
        last_hidden_states = outputs.last_hidden_state

        # Reshape back to sequence format if needed
        last_hidden_states = last_hidden_states.view(batch_size, seq_len, -1)

        return last_hidden_states

