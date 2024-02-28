import torch.nn as nn
import torchvision.models as models
import torchsummary
from torchvision.models.resnet import ResNet50_Weights

class ResNet50(nn.Module):
    def __init__(self, n_features=128):
        super().__init__()
        # Load a pre-trained (or not) ResNet-50 model
        self.resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT, num_classes=n_features)
        
    def forward(self, x):
        # X is a batch of videos with shape: [batches, sequences, channels, height, width]
        batch_size, seq_len = x.size(0), x.size(1)

        # Fuse batch and sequence dimensions before passing images through the model
        # This way each sequence of images is also considered a batch as well
        x = x.view(batch_size * seq_len, *x.shape[2:])
        
        # Feature extraction through ResNet-50
        x = self.resnet50(x)

        # Reshape back to batches of videos
        x = x.view(batch_size, seq_len, -1)

        return x

if __name__ == '__main__':
    # Print model summary
    model = ResNet50().to('cuda')
    torchsummary.summary(model, (1, 3, 224, 224))