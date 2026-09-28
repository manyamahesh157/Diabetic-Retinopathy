"""
sanjeevani_ai.models.cbam
-------------------------
Convolutional Block Attention Module (CBAM) - Woo et al., ECCV 2018.

Combines Channel Attention (WHAT features to focus on: e.g. red hemorrhage channels vs yellow exudate channels)
and Spatial Attention (WHERE to focus: e.g. foveal vs mid-peripheral retinal sectors).

Provides an architecture-native visual attention map that serves as an independent,
zero-overhead explanation mechanism to cross-validate with Grad-CAM++.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        reduced_planes = max(in_planes // ratio, 8)
        self.fc1 = nn.Conv2d(in_planes, reduced_planes, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(combined)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, planes: int, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_gate = ChannelAttention(planes, ratio)
        self.spatial_gate = SpatialAttention(kernel_size)
        self.last_spatial_map: torch.Tensor = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel Attention re-weighting
        ch_att = self.channel_gate(x)
        x = x * ch_att

        # Spatial Attention re-weighting
        sp_att = self.spatial_gate(x)
        self.last_spatial_map = sp_att.detach()  # Retained for inspection
        x = x * sp_att
        return x
