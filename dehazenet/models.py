import torch.nn as nn

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
    def __init__(self):
        super().__init__()
        self.pipeline = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            ResBlock(32), ResBlock(32),
            ResBlock(32),
            ResBlock(32), ResBlock(32),
            nn.Conv2d(32, 3, 3, 1, 1)
        )
    def forward(self, x):
        return self.pipeline(x)