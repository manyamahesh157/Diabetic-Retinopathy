"""
preprocessing.py
----------------
Fundus-photo-specific preprocessing. Generic ImageNet preprocessing (just
resize + normalize) throws away a lot of signal in retinal images because
the lesions (microaneurysms, hemorrhages, exudates) are low-contrast red/
yellow dots on a red background. We use the same pipeline that won the
Kaggle APTOS-2019 / EyePACS DR-detection competitions:

Algorithms used here:
1. Circular field-of-view crop  - removes the black border every fundus
   camera produces, so the model doesn't waste capacity on background.
2. Ben Graham's local-average-color subtraction - approximates unsharp
   masking; it removes uneven illumination and boosts local contrast of
   lesions dramatically. (This single trick is what pushed EyePACS-winning
   solutions from ~0.80 to ~0.85 quadratic-weighted-kappa.)
3. CLAHE (Contrast Limited Adaptive Histogram Equalization) on the green
   channel - the green channel carries the most vascular/lesion contrast
   in a fundus image (red channel is saturated, blue is noisy).
"""

import cv2
import numpy as np


def _crop_to_circle(img: np.ndarray) -> np.ndarray:
    """Detects the circular retinal field of view and crops the black margin."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    if w < 20 or h < 20:          # degenerate contour, skip cropping
        return img
    return img[y:y + h, x:x + w]


def _ben_graham_normalize(img: np.ndarray, sigma_frac: float = 10.0) -> np.ndarray:
    """Subtracts the local average color to flatten uneven illumination
    and sharply boost lesion contrast."""
    sigma = img.shape[1] / sigma_frac
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    normalized = cv2.addWeighted(img, 4, blurred, -4, 128)
    return normalized


def _clahe_green_channel(img: np.ndarray) -> np.ndarray:
    """Applies CLAHE to the green channel, then merges back — improves
    local contrast of microaneurysms/hemorrhages without over-amplifying noise."""
    b, g, r = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    g_eq = clahe.apply(g)
    return cv2.merge([b, g_eq, r])


def preprocess_fundus_image(img_bgr: np.ndarray, img_size: int) -> np.ndarray:
    """
    Full preprocessing pipeline. Input: raw BGR image (as read by cv2 / from
    camera). Output: (img_size, img_size, 3) float32 array in [0, 1], RGB.
    """
    img = _crop_to_circle(img_bgr)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img = _clahe_green_channel(img)
    img = _ben_graham_normalize(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_model_tensor(img_float_rgb: np.ndarray):
    """Normalizes with ImageNet stats (backbone was pretrained on ImageNet)
    and converts HWC -> CHW torch tensor."""
    import torch
    normed = (img_float_rgb - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(normed.transpose(2, 0, 1)).float()
    return tensor
