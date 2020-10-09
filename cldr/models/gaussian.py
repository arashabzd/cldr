import torch
import torch.nn as nn

from ..nets.conv import Convnet
from ..nets.mlp import MLP


class GaussianModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = Convnet()
        self.mean = nn.Linear(256, 10)
        self.logvar = nn.Linear(256, 10)
        self.p = MLP(10, [10], 10)
    
    def forward(self, x):
        dist = self.get_dist(x)
        return dist.rsample()
    
    def project(self, x):
        z = self(x)
        return z, self.p(z)

    def get_dist(self, x):
        out = self.conv(x)
        mean = self.mean(out)
        logvar = self.logvar(out)
        std = torch.exp(.5*logvar)
        dist = torch.distributions.Normal(mean, std)
        return dist