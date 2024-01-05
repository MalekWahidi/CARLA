import torch
import torch.nn as nn
import torch.nn.functional as F

from ncps.torch import CfC
from ncps.wirings import AutoNCP

class NCP_CfC(nn.Module):
    def __init__(self, n_features, n_neurons, n_outputs):
        super().__init__()
        wiring = AutoNCP(n_neurons, n_outputs)
        self.rnn = CfC(n_features, wiring, batch_first=True)

    def forward(self, x, hx=None):
        x, hx = self.rnn(x, hx)

        steer = torch.tanh(x[:, :, 0])
        throttle = torch.sigmoid(x[:, :, 1])
        brake = torch.sigmoid(x[:, :, 2])

        # Stack the 3 output values across sequences and batches
        controls = torch.stack([steer, throttle, brake], dim=2)

        return controls, hx
