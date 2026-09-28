"""
sanjeevani_ai.models.backbone
-----------------------------
Modular Feature Extractor Backbone Wrapper using timm.

Defaults to EfficientNet-B0 for high-efficiency rural screening deployment (< 25ms on CPU),
while maintaining full compatibility with B3, ResNet, or ConvNeXt backbones.
"""

from typing import Tuple
import torch
import torch.nn as nn
import timm


class RetinalBackbone(nn.Module):
    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        in_chans: int = 3,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
            global_pool="",
        )
        self.num_features = self.encoder.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns spatial feature map (B, C, H, W)."""
        return self.encoder(x)
