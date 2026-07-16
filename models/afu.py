import torch
import torch.nn as nn


class AdaptiveFeatureUnit(nn.Module):
    def __init__(self, ch, r=2, g=4):
        super().__init__()
        self.ch = ch
        self.split_ch = ch // r
        hidden_gate = max(4, ch // g)
        self.proj = nn.Conv2d(ch, self.split_ch, 1, bias=False)
        self.norm_proj = nn.BatchNorm2d(self.split_ch)
        self.refine = nn.Conv2d(self.split_ch, ch - self.split_ch, 3, padding=1, groups=self.split_ch, bias=False)
        self.norm_refine = nn.BatchNorm2d(ch - self.split_ch)
        self.gate_pool = nn.AdaptiveAvgPool2d(1)
        self.gate_reduce = nn.Conv2d(self.split_ch, hidden_gate, 1, bias=False)
        self.gate_act = nn.GELU()
        self.gate_expand = nn.Conv2d(hidden_gate, ch, 1)
        self.gate_sig = nn.Sigmoid()

    def forward(self, x):
        z = self.norm_proj(self.proj(x))
        aux = self.norm_refine(self.refine(z))
        feat = torch.cat([z, aux], dim=1)
        gate = self.gate_pool(z)
        gate = self.gate_reduce(gate)
        gate = self.gate_act(gate)
        gate = self.gate_expand(gate)
        gate = self.gate_sig(gate)
        return feat * gate
