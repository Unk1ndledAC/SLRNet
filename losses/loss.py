import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGGFeature(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights='DEFAULT').features
        self.slice = nn.Sequential(*list(vgg.children())[:16]).eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.slice(x)


vgg_feat = None


def get_vgg_feat(device):
    global vgg_feat
    if vgg_feat is None:
        vgg_feat = VGGFeature().to(device)
    return vgg_feat


def loss_perc(x, y):
    return F.mse_loss(vgg_feat(x), vgg_feat(y))


def loss_freq(x, y):
    return F.mse_loss(
        torch.fft.rfft2(x, norm='ortho').abs(),
        torch.fft.rfft2(y, norm='ortho').abs(),
    )


def loss_cont(feat_pred, feat_gt):
    f_p = F.adaptive_avg_pool2d(feat_pred, (1, 1))
    f_g = F.adaptive_avg_pool2d(feat_gt, (1, 1))
    cos_sim = F.cosine_similarity(f_p, f_g, dim=1)
    loss = 1.0 - cos_sim.clamp(min=0.0, max=1.0)
    return loss.mean()
