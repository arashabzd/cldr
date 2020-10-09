import torch
import torch.nn as nn

from ..nets.conv import Convnet
from ..nets.mlp import MLP


class DeterministicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.r = nn.Sequential(
            Convnet(),
            nn.Linear(256, 10)
        )
        self.p = MLP(10, [10], 10)
    
    def forward(self, x):
        return self.r(x)
    
    def project(self, x):
        z = self(x)
        return z, self.p(z), None
