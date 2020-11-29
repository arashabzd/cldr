import torch
import torch.nn as nn

from .utils import kaiming_init
from .utils import normal_init


class Decoder(nn.Module):
    def __init__(self, nf, dpf, activation=nn.ReLU, bn=False, init_mode=None):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(nf*dpf, 256),
            activation(),
            nn.Linear(256, 1024),
            activation(),
            nn.Unflatten(-1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            activation(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            activation(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            activation(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )
        if init_mode is not None:
            self.weight_init(init_mode)
    
    def weight_init(self, init_mode):
        if init_mode == 'kaiming':
            initializer = kaiming_init
        elif init_mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def forward(self, z):
        return self.decoder(z)