import torch
from torch import nn

from ..nets.conv import Convnet
from ..nets.mlp import MLP
from ..utils import utils


class GaussianModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = Convnet()
        self.mean = nn.Linear(256, 10)
        self.logvar = nn.Linear(256, 10)
        self.p = MLP(10, [10], 10)
    
    def reparametrize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x):
        out = self.conv(x)
        mean = self.mean(out)
        logvar = self.logvar(out)
        return self.reparametrize(mean, logvar)
    
    def project(self, x):
        return self.p(self(x))