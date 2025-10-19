import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

lambda_perc = 0.1
lambda_freq = 0.01
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VGGFeat(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights='DEFAULT').features
        self.slice = nn.Sequential(*list(vgg.children())[:16]).eval()
        for p in self.parameters():
            p.requires_grad = False
    def forward(self, x):
        return self.slice(x)

vgg_feat = VGGFeat().to(device)

def loss_perc(x, y):
    return F.mse_loss(vgg_feat(x), vgg_feat(y))

def loss_freq(x, y):
    return F.mse_loss(torch.fft.rfft2(x, norm='ortho').abs(),
                      torch.fft.rfft2(y, norm='ortho').abs())

def loss_dehaze(J_hat, J):
    l1 = F.l1_loss(J_hat, J)
    return l1 + lambda_perc * loss_perc(J_hat, J) + lambda_freq * loss_freq(J_hat, J)

"""
from .models import ResBlock

lambda_res = 0.5

class ResFeat(nn.Module):
    def __init__(self):
        super().__init__()
        self.pipeline = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            *[ResBlock(32) for _ in range(9)]
        )
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.pipeline(x)

res_feat = ResFeat().to(device)

def loss_res(x, y): 
    return F.mse_loss(res_feat(x), res_feat(y))
"""