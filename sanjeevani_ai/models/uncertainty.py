"""
sanjeevani_ai.models.uncertainty
--------------------------------
Uncertainty Quantification Engine for Clinical Screening Safety.

In ophthalmological screening, standard softmax output is well-known to be
overconfident and poorly calibrated. Sanjeevani-AI implements:
1. Monte Carlo Dropout (Gal & Ghahramani, 2016):
   - Epistemic uncertainty through stochastic dropout sampling (20 passes).
   - Predictive entropy and BALD Mutual Information.
2. Test-Time Augmentation (TTA):
   - Flips and 90-degree rotations (anatomically label-preserving for retinal fundus).
3. Tri-Tier Uncertainty Triage:
   - LOW / MEDIUM / HIGH.
   - HIGH uncertainty automatically triggers a mandatory referral flag.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class UncertaintyReport:
    uncertainty_level: str           # "LOW", "MEDIUM", "HIGH"
    predictive_entropy: float        # Total predictive uncertainty
    mutual_information: float        # Epistemic (model knowledge) uncertainty (BALD)
    mc_variance: float               # Variance across stochastic MC passes
    tta_variance: float              # Variance across spatial augmentations
    is_flagged_for_review: bool      # True if HIGH or borderline
    clinical_note: str               # Human-readable safety rationale
    pass_distributions: List[List[float]]  # Distributions across MC passes for visualization

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.uncertainty_level,
            "predictive_entropy": round(self.predictive_entropy, 3),
            "mutual_information": round(self.mutual_information, 4),
            "mc_variance": round(self.mc_variance, 4),
            "tta_variance": round(self.tta_variance, 4),
            "human_review_recommended": self.is_flagged_for_review,
            "clinical_note": self.clinical_note,
        }


def enable_mc_dropout(model: nn.Module):
    """Activates dropout layers while keeping batch norm and other layers in eval mode."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


@torch.no_grad()
def estimate_uncertainty_mc_dropout(
    model: nn.Module,
    x: torch.Tensor,
    n_passes: int = 20
) -> Tuple[torch.Tensor, float, float, float, List[List[float]]]:
    """
    Executes N stochastic forward passes using MC-Dropout.
    Returns:
       - mean_probs: (num_classes,)
       - predictive_entropy: float
       - mutual_info (BALD): float
       - max_variance: float
       - all_probs_list: list of per-pass prob lists
    """
    model.eval()
    enable_mc_dropout(model)

    pass_probs = []
    for _ in range(n_passes):
        out = model(x)
        pass_probs.append(out["probs"].squeeze(0))

    stacked = torch.stack(pass_probs, dim=0)  # (N, C)
    mean_probs = stacked.mean(dim=0)          # (C,)

    # 1. Total Predictive Entropy: H(p_mean)
    eps = 1e-9
    predictive_entropy = float(-torch.sum(mean_probs * torch.log(mean_probs + eps)).item())

    # 2. Expected Entropy across passes: E[H(p_t)]
    per_pass_entropy = -torch.sum(stacked * torch.log(stacked + eps), dim=1).mean().item()

    # 3. Mutual Information (BALD proxy for epistemic uncertainty): H(p_mean) - E[H(p_t)]
    mutual_info = float(max(0.0, predictive_entropy - per_pass_entropy))

    # 4. Variance across passes
    variances = stacked.var(dim=0)
    max_variance = float(variances.max().item())

    all_probs_list = stacked.cpu().numpy().tolist()

    return mean_probs, predictive_entropy, mutual_info, max_variance, all_probs_list


@torch.no_grad()
def predict_with_tta(model: nn.Module, x: torch.Tensor, n_variants: int = 5) -> Tuple[torch.Tensor, float]:
    """
    Test-Time Augmentation on 5 spatial rotations and reflections.
    Retinal fundus has no canonical orientation for DR grade diagnosis.
    """
    model.eval()
    variants = [
        x,
        torch.flip(x, dims=[3]),                  # Horizontal flip
        torch.flip(x, dims=[2]),                  # Vertical flip
        torch.rot90(x, k=1, dims=[2, 3]),         # 90 deg rotation
        torch.rot90(x, k=3, dims=[2, 3]),         # 270 deg rotation
    ][:n_variants]

    probs_list = []
    for v in variants:
        out = model(v)
        probs_list.append(out["probs"].squeeze(0))

    stacked = torch.stack(probs_list, dim=0)  # (V, C)
    mean_probs = stacked.mean(dim=0)
    tta_variance = float(stacked.var(dim=0).max().item())

    return mean_probs, tta_variance


def quantify_prediction_uncertainty(
    model: nn.Module,
    x: torch.Tensor,
    n_mc_passes: int = 20,
    n_tta_variants: int = 5,
    entropy_low: float = 0.50,
    entropy_high: float = 0.85,
    variance_low: float = 0.03,
) -> Tuple[torch.Tensor, UncertaintyReport]:
    """
    Runs combined MC-Dropout and TTA uncertainty evaluation.
    Returns:
       - calibrated_probs: Tensor (num_classes,)
       - uncertainty_report: UncertaintyReport
    """
    # 1. MC-Dropout Sampling
    mc_probs, pred_entropy, mutual_info, mc_var, mc_history = estimate_uncertainty_mc_dropout(
        model, x, n_passes=n_mc_passes
    )

    # 2. Test-Time Augmentation
    tta_probs, tta_var = predict_with_tta(model, x, n_variants=n_tta_variants)

    # 3. Blended Calibrated Probability
    fused_probs = 0.60 * mc_probs + 0.40 * tta_probs

    # 4. Uncertainty Level Categorization
    if pred_entropy < entropy_low and mc_var < variance_low and tta_var < variance_low:
        level = "LOW"
        flag = False
        note = "Model prediction is stable across stochastic sampling passes and spatial orientations."
    elif pred_entropy >= entropy_high or mc_var > 0.06 or mutual_info > 0.15:
        level = "HIGH"
        flag = True
        note = (
            "Significant variance observed across stochastic passes or orientations. "
            "Case exhibits borderline or ambiguous lesion features; mandatory clinical confirmation required."
        )
    else:
        level = "MEDIUM"
        flag = False
        note = "Moderate prediction dispersion. Correlate with clinical history and lesion findings."

    report = UncertaintyReport(
        uncertainty_level=level,
        predictive_entropy=pred_entropy,
        mutual_information=mutual_info,
        mc_variance=mc_var,
        tta_variance=tta_var,
        is_flagged_for_review=flag,
        clinical_note=note,
        pass_distributions=mc_history,
    )

    return fused_probs, report
