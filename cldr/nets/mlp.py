import torch
import torch.nn as nn
from .utils import kaiming_init, normal_init


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, 
                 activation=nn.ReLU, 
                 bn=False,
                 init_mode=None):
        super().__init__()
        features = [in_features] + hidden_features
        layers = []
        for i in range(len(features) - 1):
            layers.append(nn.Linear(features[i], features[i+1], bias=not bn))
            if bn:
                layers.append(nn.BatchNorm1d(features[i+1]))
            layers.append(activation())
        layers.append(nn.Linear(features[-1], out_features))
        self.mlp = nn.Sequential(*layers)
        
        if init_mode is not None:
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
        return self.mlp(x)