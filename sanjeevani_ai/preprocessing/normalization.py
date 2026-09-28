"""
sanjeevani_ai.preprocessing.normalization
-----------------------------------------
Color, Illumination, and Contrast Normalization for Retinal Fundus Images.

Techniques:
1. Ben Graham's Local Color Normalization:
   Retinal images from different fundus cameras exhibit widely different illumination,
   vignetting, and pigmentations. Local Gaussian blur subtraction:
       I_norm = alpha * I + beta * GaussianBlur(I) + gamma
   equalizes background gradients while accentuating microaneurysms and exudates.
2. Green-Channel CLAHE:
   Hemoglobin absorption is strongest at 570nm (green spectrum). CLAHE enhances
   microaneurysms, hemorrhages, and neovascular shunts without blowing out noise.
"""

from typing import Tuple
import cv2
import numpy as np
import torch


def apply_ben_graham_normalization(
    img_rgb: np.ndarray,
    sigma: int = 10,
    alpha: float = 4.0,
    beta: float = -4.0,
    gamma: float = 128.0
) -> np.ndarray:
    """
    Implements Ben Graham's local average color subtraction.
    Input: uint8 RGB image [0, 255].
    Output: uint8 RGB image [0, 255].
    """
    blurred = cv2.GaussianBlur(img_rgb, (0, 0), sigmaX=sigma)
    normalized = cv2.addWeighted(img_rgb, alpha, blurred, beta, gamma)
    return np.clip(normalized, 0, 255).astype(np.uint8)


def apply_green_channel_clahe(
    img_rgb: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Enhances microvascular contrast by applying CLAHE to the green channel.
    Input: uint8 RGB image.
    Output: uint8 RGB image with enhanced vessel and lesion definition.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    r, g, b = cv2.split(img_rgb)
    g_enhanced = clahe.apply(g)
    return cv2.merge([r, g_enhanced, b])


def normalize_to_tensor(
    img_rgb_float: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> torch.Tensor:
    """
    Converts float RGB image in [0, 1] (H, W, 3) to standard PyTorch tensor (3, H, W).
    """
    tensor = torch.from_numpy(img_rgb_float.transpose((2, 0, 1))).float()
    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)
    tensor = (tensor - mean_tensor) / std_tensor
    return tensor
