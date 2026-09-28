"""
model.py
--------
Architecture + training losses + uncertainty estimation.

Algorithms used:
1. Transfer learning on EfficientNet-B3 (pretrained on ImageNet via `timm`).
   EfficientNet is chosen over plain ResNet because its compound scaling
   gives a strictly better accuracy/FLOPs trade-off — good for a laptop-
   deployable hackathon demo.
2. CBAM-style channel+spatial attention block inserted before the
   classifier head — this both (a) improves accuracy by letting the net
   re-weight lesion-relevant channels/regions, and (b) gives us a SECOND,
   architecture-native form of explainability (the attention map) that is
   independent of Grad-CAM, so we can cross-check the two explanations.
3. Ordinal-aware loss: DR grades are ordinal (0..4), so instead of vanilla
   cross-entropy we use a Focal Loss (handles class imbalance — grade 0
   dominates real datasets) combined with an ordinal regression penalty
   (CORAL-style: penalize predictions that are numerically far from the
   true grade more than adjacent-grade confusions).
4. Monte-Carlo Dropout for uncertainty: at inference we keep dropout ON
   and run N stochastic forward passes; the variance across passes gives
   a principled uncertainty estimate (Gal & Ghahramani, 2016) — used to
   flag low-confidence cases for human specialist review.
5. Test-Time Augmentation (TTA): average predictions over flips/rotations
   of the same image to reduce variance and squeeze out extra accuracy
   with zero extra training cost.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from . import config


class CBAM(nn.Module):
    """Convolutional Block Attention Module: channel attention then spatial
    attention. Cheap (few extra params) and gives an inspectable attention map."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        ch_att = torch.sigmoid(self.channel_fc(x))
        x = x * ch_att

        avg_pool = x.mean(dim=1, keepdim=True)
        max_pool, _ = x.max(dim=1, keepdim=True)
        sp_att = torch.sigmoid(self.spatial_conv(torch.cat([avg_pool, max_pool], dim=1)))
        x = x * sp_att
        self.last_attention_map = sp_att.detach()   # stashed for explainability
        return x


class DRGradingModel(nn.Module):
    def __init__(self, backbone_name: str = config.BACKBONE,
                 num_classes: int = config.NUM_CLASSES,
                 dropout_p: float = config.DROPOUT_P,
                 pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool=""
        )
        feat_channels = self.backbone.num_features
        self.attention = CBAM(feat_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(feat_channels, num_classes)

    def forward(self, x):
        feats = self.backbone.forward_features(x)   # (B, C, H, W)
        feats = self.attention(feats)
        pooled = self.pool(feats).flatten(1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

    def get_last_conv_feature_map(self, x):
        """Used by Grad-CAM: returns spatial feature map before pooling."""
        return self.attention(self.backbone.forward_features(x))


# ----------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = config.FOCAL_GAMMA, class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, logits, targets):
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        logp_t = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -((1 - p_t) ** self.gamma) * logp_t
        if self.class_weights is not None:
            w = self.class_weights.to(logits.device)[targets]
            loss = loss * w
        return loss.mean()


def ordinal_penalty(logits, targets, num_classes=config.NUM_CLASSES):
    """Extra penalty proportional to |predicted_grade - true_grade|, so a
    confusion between grade 0 and grade 4 is punished harder than between
    grade 2 and grade 3 (clinically far more serious mistake)."""
    probs = F.softmax(logits, dim=1)
    grades = torch.arange(num_classes, device=logits.device).float()
    expected_grade = (probs * grades).sum(dim=1)
    return F.mse_loss(expected_grade, targets.float())


def combined_loss(logits, targets, class_weights=None, ordinal_weight=0.3):
    fl = FocalLoss(class_weights=class_weights)(logits, targets)
    op = ordinal_penalty(logits, targets)
    return fl + ordinal_weight * op


# ----------------------------------------------------------------------
# Uncertainty-aware inference
# ----------------------------------------------------------------------
def enable_mc_dropout(model: nn.Module):
    """Keeps dropout layers active during eval() for MC-Dropout sampling."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def predict_with_uncertainty(model: nn.Module, x: torch.Tensor,
                              n_passes: int = config.MC_DROPOUT_PASSES):
    """
    Runs N stochastic forward passes (MC-Dropout) and returns:
      - mean_probs: averaged class probabilities  (B, num_classes)
      - predictive_entropy: uncertainty of the mean prediction (B,)
      - mutual_information: epistemic-uncertainty proxy (B,)
        (BALD approximation: predictive_entropy - avg(per-pass entropy))
    """
    model.eval()
    enable_mc_dropout(model)

    all_probs = []
    for _ in range(n_passes):
        logits = model(x)
        all_probs.append(F.softmax(logits, dim=1))
    stacked = torch.stack(all_probs, dim=0)          # (N, B, C)

    mean_probs = stacked.mean(dim=0)
    predictive_entropy = -(mean_probs * torch.log(mean_probs + 1e-9)).sum(dim=1)
    per_pass_entropy = -(stacked * torch.log(stacked + 1e-9)).sum(dim=2).mean(dim=0)
    mutual_information = predictive_entropy - per_pass_entropy

    return mean_probs, predictive_entropy, mutual_information


@torch.no_grad()
def predict_with_tta(model: nn.Module, x: torch.Tensor):
    """Test-Time Augmentation: average predictions over horizontal/vertical
    flips and 90-degree rotations of the same image (retinal images have no
    canonical orientation dependency for grading, so these are label-preserving)."""
    model.eval()
    variants = [
        x,
        torch.flip(x, dims=[3]),                  # horizontal flip
        torch.flip(x, dims=[2]),                  # vertical flip
        torch.rot90(x, k=1, dims=[2, 3]),
        torch.rot90(x, k=3, dims=[2, 3]),
    ]
    probs = [F.softmax(model(v), dim=1) for v in variants[:config.TTA_VARIANTS]]
    return torch.stack(probs, dim=0).mean(dim=0)
