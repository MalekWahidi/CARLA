import torch
import torch.nn as nn
import torch.nn.functional as F

from ncps.torch import CfC
from ncps.wirings import AutoNCP

class Cond_NCP(nn.Module):
    def __init__(self, n_features, n_neurons, n_outputs, cell_type="cfc"):
        super().__init__()
        self.n_outputs = n_outputs
        wiring = AutoNCP(n_neurons, n_outputs, seed=0)

        if cell_type == "ltc":
            self.rnn = LTC(n_features, wiring, batch_first=True)
        else:
            self.rnn = CfC(n_features, wiring, batch_first=True)

    def forward(self, features, command, hx=None):
        # Concatenate image features and high-level command along feature dim
        x = torch.cat((features, command), dim=-1)

        x, hx = self.rnn(x, hx)

        steer = torch.tanh(x[:, :, 0])
        throttle = torch.sigmoid(x[:, :, 1])
        brake = torch.sigmoid(x[:, :, 2])

        # Stack the 3 output values across sequences and batches
        controls = torch.stack([steer, throttle, brake], dim=2)

        return controls, hx
