import torch.nn as nn

class ConvHead(nn.Module):
    def __init__(self, n_features=64):
        super().__init__()
        # Sequential block for convolutional and batch normalization layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 7, padding=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 7, padding=2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 5, padding=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 5, padding=2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, padding=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 3, padding=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_features, 3, padding=2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(n_features)
        )

    def forward(self, x):
        # X is a batch of videos with shape: [batches, sequences, channels, height, width]
        batch_size, seq_len = x.size(0), x.size(1)
        # Fuse batch and sequence dimensions before passing images through the model
        # This way each sequence of images is also considered a batch as well
        x = x.view(batch_size * seq_len, *x.shape[2:])
                
        # Apply convolutional operations
        x = self.conv_layers(x)

        # Global average pooling
        x = x.mean((-1, -2))

        # Reshape back to batches of videos
        x = x.view(batch_size, seq_len, -1)

        return x
