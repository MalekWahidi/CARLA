import torch
import torch.nn as nn

from ncps.torch import CfC, LTC
from ncps.wirings import AutoNCP

class NCP(nn.Module):
    def __init__(self, n_features, n_neurons, n_outputs, cell_type="cfc"):
        super().__init__()
        self.n_outputs = n_outputs
        wiring = AutoNCP(n_neurons, n_outputs, seed=0)

        if cell_type == "cfc":
            self.rnn = CfC(n_features, wiring, batch_first=True)
        elif cell_type == "ltc":
            self.rnn = LTC(n_features, wiring, batch_first=True)
        else:
            print("Invalid cell type for NCP model")

    def forward(self, x, hx=None):
        x, hx = self.rnn(x, hx)

        # In case of 3 outputs (steer, throttle, brake)
        if self.n_outputs == 3:
            steer = torch.tanh(x[:, :, 0])
            throttle = torch.sigmoid(x[:, :, 1])
            brake = torch.sigmoid(x[:, :, 2])
            controls = torch.stack([steer, throttle, brake], dim=2)

        # In case of 1 output (steer only)
        elif self.n_outputs == 1:
            steer = torch.tanh(x[:, :, 0])
            controls = steer.unsqueeze(-1)  # Add an extra dimension for consistency

        return controls, hx