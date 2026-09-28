"""
sanjeevani_ai.xai.agreement
---------------------------
Multi-Method Explanation Agreement & Trust Verification (Research Metric).

Standard medical AI showcases a single Grad-CAM heatmap without verification.
Sanjeevani-AI measures spatial consensus between independent attribution mechanisms:
1. Grad-CAM++ (Gradient-weighted coarse feature map activations)
2. Integrated Gradients (Axiomatic pixel-level path integral attributions)
3. CBAM Spatial Attention (Architecture-native feature re-weighting)

Calculates:
- Hotspot Intersection-over-Union (IoU)
- Continuous spatial Pearson correlation
- Explanation Agreement Score [0.0 - 1.0]

IMPORTANT:
This is a proposed research metric for explanation consistency and has not been
clinically validated as a diagnostic surrogate.
"""

from typing import Dict, Any, Optional
import numpy as np
import cv2


def compute_explanation_agreement(
    cam_map: np.ndarray,
    ig_map: np.ndarray,
    cbam_map: Optional[np.ndarray] = None,
    percentile_threshold: float = 75.0
) -> Dict[str, Any]:
    """
    Computes spatial consensus between Grad-CAM++ and Integrated Gradients.
    Returns:
        - agreement_score: float in [0.0, 1.0]
        - iou: float in [0.0, 1.0]
        - spatial_correlation: float in [-1.0, 1.0]
        - status: "HIGH_CONSENSUS", "MODERATE", "DIVERGENT"
        - recommendation: str
    """
    # Resize to identical shape if necessary
    h, w = cam_map.shape[:2]
    if ig_map.shape[:2] != (h, w):
        ig_map = cv2.resize(ig_map, (w, h), interpolation=cv2.INTER_LINEAR)

    # 1. Hotspot Binarization at upper quantile
    cam_thresh = np.percentile(cam_map, percentile_threshold)
    ig_thresh = np.percentile(ig_map, percentile_threshold)

    cam_bin = (cam_map >= cam_thresh).astype(bool)
    ig_bin = (ig_map >= ig_thresh).astype(bool)

    # 2. Intersection over Union (IoU)
    intersection = np.logical_and(cam_bin, ig_bin).sum()
    union = np.logical_or(cam_bin, ig_bin).sum()
    iou = float(intersection / (union + 1e-8))

    # 3. Continuous Pearson Spatial Correlation
    cam_flat = cam_map.flatten()
    ig_flat = ig_map.flatten()
    if np.std(cam_flat) > 1e-6 and np.std(ig_flat) > 1e-6:
        corr_matrix = np.corrcoef(cam_flat, ig_flat)
        corr = float(corr_matrix[0, 1])
    else:
        corr = 0.0

    # 4. Composite Agreement Score
    # IoU is scaled, correlation is clipped to [0, 1]
    norm_corr = max(0.0, corr)
    composite_score = float(np.clip(0.60 * iou + 0.40 * norm_corr, 0.0, 1.0))

    # 5. Status & Clinical Warning
    if composite_score >= 0.45:
        status = "HIGH_CONSENSUS"
        note = "Independent attribution methods corroborate the same primary retinal lesion locations."
        flag = False
    elif composite_score >= 0.25:
        status = "MODERATE_CONSENSUS"
        note = "Moderate spatial overlap between coarse activation and pixel-level attribution."
        flag = False
    else:
        status = "DIVERGENT"
        note = (
            "⚠ Explanation divergence detected: Grad-CAM++ and Integrated Gradients identify conflicting "
            "retinal zones. Explanation reliability is reduced; prioritize human ophthalmic examination."
        )
        flag = True

    return {
        "agreement_score": round(composite_score, 3),
        "iou": round(iou, 3),
        "correlation": round(corr, 3),
        "status": status,
        "is_flagged": flag,
        "clinical_note": note,
    }
