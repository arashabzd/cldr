import torch
import torch.nn as nn

from .utils import kaiming_init
from .utils import normal_init


class Encoder(nn.Module):
    def __init__(self, activation=nn.ReLU, bn=False, init_mode=None):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=not bn),
            nn.BatchNorm2d(32) if bn else nn.Identity(),
            activation(),
            nn.Conv2d(32, 32, 4, 2, 1, bias=not bn),
            nn.BatchNorm2d(32) if bn else nn.Identity(),
            activation(),
            nn.Conv2d(32, 64, 2, 2, 1, bias=not bn),
            nn.BatchNorm2d(64) if bn else nn.Identity(),
            activation(),
            nn.Conv2d(64, 64, 2, 2, 1, bias=not bn),
            nn.BatchNorm2d(64) if bn else nn.Identity(),
            activation(),
            nn.Flatten(),
            nn.Linear(1600, 256),
            activation()
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

    def forward(self, x):
        return self.encoder(x)