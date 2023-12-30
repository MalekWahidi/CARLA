import torch.nn as nn
import torch.nn.functional as F

class ConvHead(nn.Module):
    def __init__(self, n_features=64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 7, padding=2, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 7, padding=2, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2, stride=2)
        self.conv4 = nn.Conv2d(64, 128, 5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=2, stride=2)
        self.conv6 = nn.Conv2d(256, 128, 3, padding=2, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 64, 3, padding=2, stride=2)
        self.conv8 = nn.Conv2d(64, n_features, 3, padding=2, stride=2)
        self.bn4 = nn.BatchNorm2d(n_features)

    def forward(self, x):
        # Reshaping assuming x is a batch of sequences of images
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.view(batch_size * seq_len, *x.shape[2:])
                
        # Convolutional operations with ReLU activations and batch normalization
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.bn1(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.bn2(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.bn3(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.bn4(x))

        # Global average pooling
        x = x.mean((-1, -2))

        # Reshape back to sequence format if needed
        x = x.view(batch_size, seq_len, -1)

        return x
