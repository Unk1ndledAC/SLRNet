import torch.nn as nn
import math, torch

class SEModule(nn.Module):
    def __init__(self, ch=32, red=4):
        super().__init__()
        self.pipeline = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch//red, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(ch//red, ch, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.pipeline(x)

class GhostModule(nn.Module):
    def __init__(self, inp, oup, ratio=2):
        super().__init__()
        self.oup = oup
        ch1 = math.ceil(oup / ratio)
        ch2 = ch1 * (ratio - 1)

        self.pipeline1 = nn.Sequential(
            nn.Conv2d(inp, ch1, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ch1),
            nn.ReLU(True),
        )
        self.pipeline2 = nn.Sequential(
            nn.Conv2d(ch1, ch2, 3, 1, 1, groups=ch1, bias=False),
            nn.BatchNorm2d(ch2),
            nn.ReLU(True),
        )

    def forward(self, x):
        x1 = self.pipeline1(x)
        x2 = self.pipeline2(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]

class ResBlock(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.pipeline = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1), 
            nn.ReLU(True),
            nn.Conv2d(ch, ch, 3, 1, 1)
        )
    def forward(self, x): 
        return x + self.pipeline(x)

class DehazeNet(nn.Module):
    def __init__(self, ch=32, n_gse=5, n_res=5):
        super().__init__()
        self.pipeline = nn.Sequential(
            nn.Conv2d(3, ch, 3, 1, 1),
            *[m for _ in range(n_gse) for m in (GhostModule(ch, ch), SEModule(ch))],
            *[ResBlock(ch) for _ in range(n_res)],
            nn.Conv2d(ch, 3, 3, 1, 1)
        )

    def forward(self, x):
        return self.pipeline(x)
        

