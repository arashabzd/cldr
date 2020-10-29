import torch
import torch.nn as nn

from ..nets.mlp import MLP


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.d = MLP(
            in_features=10, 
            hidden_features=[1000]*6, 
            out_features=2,
            activation=nn.LeakyReLU
        )

    def forward(self, z):
        return self.d(z)
