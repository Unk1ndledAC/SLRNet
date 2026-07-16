import torch
import torch.nn as nn
import torch.nn.functional as F

from .afu import AdaptiveFeatureUnit


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + res


class SLRNet(nn.Module):
    def __init__(self, base_ch=32, n_blocks=5):
        super().__init__()
        self.in_proj = nn.Conv2d(3, base_ch, 3, padding=1)
        self.enhancer = AdaptiveFeatureUnit(base_ch)
        self.backbone = nn.ModuleList([ResBlock(base_ch) for _ in range(n_blocks)])
        self.refiner = AdaptiveFeatureUnit(base_ch)
        self.out_proj = nn.Conv2d(base_ch, 3, 3, padding=1)

    def forward(self, x):
        x = self.in_proj(x)
        early_feat = self.enhancer(x)
        feat = early_feat
        for block in self.backbone:
            feat = block(feat)
        late_feat = self.refiner(feat)
        fused = late_feat + early_feat
        out = self.out_proj(fused)
        return torch.clamp(out, 0.0, 1.0)
