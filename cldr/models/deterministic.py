import torch
import torch.nn as nn

from ..nets.conv import Convnet
from ..nets.mlp import MLP


class DeterministicModel(nn.Module):
    def __init__(self, bn=False, init_mode='kaiming'):
        super().__init__()
        self.conv = Convnet(bn=bn, init_mode=init_mode)
        self.mean = nn.Linear(256, 10)
        self.head = MLP(10, [10], 10, bn=bn, init_mode=init_mode)
    
    def forward(self, x):
        return self.mean(self.conv(x))
    
    def project(self, x):
        out = self.conv(x)
        mean = self.mean(out)
        std = torch.zeros_like(mean)
        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()
        u = self.head(z)
        return [z, u, dist]
