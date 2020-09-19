import torch
import torch.nn as nn
from .utils import kaiming_init, normal_init


class Convnet(nn.Module):
    def __init__(self, bn=False, init_mode='kaiming'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=not bn),
            nn.BatchNorm2d(32) if bn else nn.Identity(),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1, bias=not bn),
            nn.BatchNorm2d(32) if bn else nn.Identity(),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=not bn),
            nn.BatchNorm2d(64) if bn else nn.Identity(),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1, bias=not bn),
            nn.BatchNorm2d(64) if bn else nn.Identity(),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1, bias=not bn),
            nn.BatchNorm2d(256) if bn else nn.Identity(),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256) if bn else nn.Identity(),
            nn.ReLU(True)
        )
        
        self.weight_init(init_mode)
    
    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def forward(self, x):
        return self.net(x)