import torch, torch.nn as nn, math

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, stride, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]
    
class SEModule(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class ResBlock(nn.Module):
    def __init__(self, ch=32):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return residual + out

class DehazeNet(nn.Module):
    def __init__(self):
        super(DehazeNet, self).__init__()

        self.head_conv = nn.Conv2d(3, 32, 3, 1, 1)

        self.ghost1 = GhostModule(32, 32)
        self.se1 = SEModule(32)
        self.ghost2 = GhostModule(32, 32)
        self.se2 = SEModule(32)
        self.ghost3 = GhostModule(32, 32)
        self.se3 = SEModule(32)
        self.ghost4 = GhostModule(32, 32)
        self.se4 = SEModule(32)
        self.ghost5 = GhostModule(32, 32)
        self.se5 = SEModule(32)     
        
        self.res1 = ResBlock(32)
        self.res2 = ResBlock(32)
        self.res3 = ResBlock(32)
        self.res4 = ResBlock(32)
        self.res5 = ResBlock(32)

        self.tail_conv = nn.Conv2d(32, 3, 3, 1, 1)

    def forward(self, x):
        feat = self.head_conv(x)  
        feat = self.ghost1(feat)     
        feat = self.se1(feat)     
        feat = self.ghost2(feat)     
        feat = self.se2(feat)  
        feat = self.ghost3(feat)     
        feat = self.se3(feat)  
        feat = self.ghost4(feat)     
        feat = self.se4(feat)  
        feat = self.ghost5(feat)     
        feat = self.se5(feat)  
           
        feat = self.res1(feat)       
        feat = self.res2(feat)
        feat = self.res3(feat)
        feat = self.res4(feat)
        feat = self.res5(feat)
        out = self.tail_conv(feat)  
        return out