"""
sanjeevani_ai.preprocessing.crop
--------------------------------
Automatic Circular Retinal Field-of-View (FOV) Detection and Border Cropping.

Removes redundant black camera margins to standardize the aspect ratio and
ensure maximum resolution of the active retinal tissue.
"""

from typing import Tuple
import cv2
import numpy as np


def crop_to_retinal_fov(
    img_bgr: np.ndarray,
    threshold_val: int = 10,
    margin_padding: int = 4
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    Finds the largest connected circular retinal field, extracts its bounding box,
    and returns (cropped_image, circular_mask, bbox).

    Bbox format: (x, y, w, h)
    """
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr, np.zeros((1, 1), dtype=np.uint8), (0, 0, 0, 0)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Binary threshold to segment active retinal tissue from dark camera boundary
    _, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)

    # Morphological closing to fill vessels and optic cup dips
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback: full frame
        mask = np.ones((h, w), dtype=np.uint8) * 255
        return img_bgr.copy(), mask, (0, 0, w, h)

    # Largest contour is the retinal circle
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(largest_contour)

    # Safety check: if detected contour is unreasonably small (< 10% of frame), don't over-crop
    if (cw * ch) < (0.10 * h * w):
        mask = np.ones((h, w), dtype=np.uint8) * 255
        return img_bgr.copy(), mask, (0, 0, w, h)

    # Apply small padding
    x_start = max(0, x - margin_padding)
    y_start = max(0, y - margin_padding)
    x_end = min(w, x + cw + margin_padding)
    y_end = min(h, y + ch + margin_padding)

    cropped_img = img_bgr[y_start:y_end, x_start:x_end]

    # Create smooth circular mask for the cropped region
    cropped_gray = gray[y_start:y_end, x_start:x_end]
    cropped_mask = np.zeros_like(cropped_gray, dtype=np.uint8)
    center = ((x_end - x_start) // 2, (y_end - y_start) // 2)
    radius = min((x_end - x_start) // 2, (y_end - y_start) // 2)
    cv2.circle(cropped_mask, center, radius, 255, -1)

    return cropped_img, cropped_mask, (x_start, y_start, x_end - x_start, y_end - y_start)
