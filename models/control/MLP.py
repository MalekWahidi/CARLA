import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_Net(nn.Module):
    def __init__(self, n_features, hidden_size, n_outputs):
        super(MLP_Net, self).__init__()
        # Define the layers of the MLP
        self.fc1 = nn.Linear(n_features, hidden_size)  # First hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Second hidden layer
        self.output_layer = nn.Linear(hidden_size, n_outputs)  # Output layer

    def forward(self, x):
        # Flatten the input if necessary
        x = x.view(x.size(0), -1)
        
        # Apply the layers with non-linear activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output_layer(x)

        # Apply specific activation functions for each control
        steer = torch.tanh(x[:, 0])
        throttle = torch.sigmoid(x[:, 1])
        brake = torch.sigmoid(x[:, 2])

        # Stack the 3 output values
        controls = torch.stack([steer, throttle, brake], dim=1)
        return controls

# Example usage:
# mlp_model = MLP_Net(n_features=10, hidden_size=20, n_outputs=3)
# input_tensor = torch.randn(5, 10)  # (batch_size, n_features)
# output = mlp_model(input_tensor)
