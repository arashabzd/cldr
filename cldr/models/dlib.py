import torch
from torch import nn

from ..nets.conv import Convnet
from ..nets.mlp import MLP


class DlibModel(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.convnet = Convnet(in_channels)
        self.mlp = MLP(256, [], 10)
    def forward(self, x):
        return self.mlp(self.convnet(x))