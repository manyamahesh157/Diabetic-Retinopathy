"""
sanjeevani_ai.xai.loupe
-----------------------
Interactive 4x Digital Ophthalmic Lesion Loupe & Microvascular Profile Analyzer.

Allows ophthalmologists to inspect suspicious microaneurysms, hemorrhages, or
neovascular tufts at 4x optical magnification with green-channel hemoglobin contrast.
"""

from typing import Dict, Any, Tuple
import cv2
import numpy as np


def extract_digital_loupe(
    img_rgb: np.ndarray,
    center_xy: Tuple[int, int],
    patch_size: int = 64,
    zoom_factor: int = 4
) -> Dict[str, Any]:
    """
    Extracts a high-magnification zoomed inspection patch around center_xy.
    Returns:
    - zoomed_rgb: np.ndarray (patch_size * zoom_factor, patch_size * zoom_factor, 3)
    - green_channel_contrast: np.ndarray (single channel enhanced)
    - vessel_intensity_profile: 1D array across the horizontal transect of the lesion
    """
    h, w = img_rgb.shape[:2]
    cx, cy = center_xy

    half = patch_size // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)

    patch = img_rgb[y1:y2, x1:x2]

    # Resize with high-order Lanczos / Bicubic interpolation
    target_dim = patch_size * zoom_factor
    zoomed_rgb = cv2.resize(patch, (target_dim, target_dim), interpolation=cv2.INTER_CUBIC)

    # Green-channel hemoglobin contrast enhancement
    g_chan = zoomed_rgb[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g_enhanced = clahe.apply(g_chan)

    # Microvascular Transect Profile (Horizontal line through center of lesion)
    mid_row = target_dim // 2
    intensity_profile = g_enhanced[mid_row, :].tolist()

    return {
        "center_xy": center_xy,
        "zoomed_rgb": zoomed_rgb,
        "green_enhanced": g_enhanced,
        "transect_profile": intensity_profile,
        "magnification": f"{zoom_factor}x Digital Ophthalmic Loupe",
    }
