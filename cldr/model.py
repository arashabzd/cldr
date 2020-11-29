import torch
import torch.nn as nn

from .nn import Encoder
from .nn import Decoder
from .nn import MLP


class MultiHeadAE(nn.Module):
    def __init__(self, nf, dpf, head_layers=[], 
                 decoder=True,
                 activation=nn.ReLU, 
                 bn=False, 
                 init_mode=None):
        super().__init__()
        self.nf = nf
        self.dpf = dpf
        self.encoder = Encoder(activation, bn, init_mode)
        self.heads = nn.ModuleList(
            [
                MLP(256, head_layers, dpf, activation, bn, init_mode) 
                for n in range(nf)
            ]
        )
        if decoder:
            self.decoder = Decoder(nf, dpf, activation, bn, init_mode)
        else:
            self.decoder = None
    
    def encode(self, x):
        c = self.encoder(x)
        z = [head(c) for head in self.heads]
        return torch.cat(z, dim=1)
    
    def decode(self, z):
        if self.decoder:
            return self.decoder(z)
        return None
    
    def forward(self, x):
        return self.encode(x)
    
    def autoencode(self, x):
        z = self.encode(x)
        xr = self.decode(z)
        return x, z, xr
