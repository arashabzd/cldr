import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_metric_learning import distances, losses


class TripletMarginLoss(nn.Module):
    def __init__(self, nf, dpf, margin=0.05):
        super().__init__()
        self.nf = nf
        self.dpf = dpf
        distance = distances.LpDistance(p=2, power=1, normalize_embeddings=False)
        self.loss = losses.TripletMarginLoss(margin=margin, distance=distance)
    
    def forward(self, z, y, f):
        loss = self.loss(z[f==0, :self.dpf], y[f==0])
        for i in range(1, self.nf):
            loss += self.loss(z[f==i, self.dpf*i:self.dpf*i+self.dpf], y[f==i])
        return loss

class SWD(nn.Module):
    def __init__(self, sampler, np=None):
        super().__init__()
        self.sampler = sampler
        self.np = np
    
    def forward(self, z):
        n, d = z.shape
        np = self.np if self.np is not None else d
        p = self.sampler(z)
        W = z.new_empty(size=(d, np))
        nn.init.orthogonal_(W)
        z = torch.matmul(z, W)
        p = torch.matmul(p, W)
        z = torch.sort(z, dim=0)[0]
        p = torch.sort(p, dim=0)[0]
        loss = F.mse_loss(z, p)
        return loss