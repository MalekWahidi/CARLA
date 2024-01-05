import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_Net(nn.Module):
    def __init__(self, n_features, n_neurons, n_outputs):
        super().__init__()

        # Initialize the LSTM layer
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=n_neurons, batch_first=True)
        
        # Linear layer to map from LSTM to controls
        self.output_layer = nn.Linear(n_neurons, n_outputs)

    def forward(self, x, hx=None):
        # x shape: (batch, sequence, features)
        x, (h_n, c_n) = self.lstm(x, hx)
        x = self.output_layer(x)

        # Apply activation functions
        steer = torch.tanh(x[:, :, 0])
        throttle = torch.sigmoid(x[:, :, 1])
        brake = torch.sigmoid(x[:, :, 2])

        controls = torch.stack([steer, throttle, brake], dim=2)

        return controls, (h_n, c_n)
