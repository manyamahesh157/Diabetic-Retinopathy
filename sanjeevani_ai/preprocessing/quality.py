"""
sanjeevani_ai.preprocessing.quality
-----------------------------------
Autonomous Clinical Quality Gate & Gradability Assessor.

In accordance with clinical screening standards and FDA-cleared autonomous
DR systems (e.g. IDx-DR/LumineticsCore), any fundus photograph with inadequate
clarity, incorrect anatomy, or severe artifacts MUST BE REJECTED before reaching
the classification or explainability models.

Checks implemented:
1. Blur Check: Variance of the Laplacian of the grayscale retina.
2. Exposure Check: Mean grayscale intensity band [min_brightness, max_brightness].
3. Glare/Specular Reflection: Percentage of pixels saturating at > 245.
4. Anatomical FOV: Retinal field boundary contour detection to verify eye presence.
5. Chromaticity / External Eye Filter: Retinal choroidal hemoglobin imparts a high
   Red-to-Blue ratio (R/B > 1.65). External ocular photos (cornea, iris, eyelids)
   exhibit balanced RGB and are flagged immediately with specific instructions.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import cv2
import numpy as np


@dataclass
class QualityReport:
    is_gradable: bool
    quality_score: float
    blur_score: float
    mean_brightness: float
    glare_percentage: float
    fov_fraction: float
    rb_ratio: float
    reasons: List[str] = field(default_factory=list)
    technician_actions: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gradable": self.is_gradable,
            "quality_score": round(self.quality_score, 3),
            "blur_score": round(self.blur_score, 1),
            "mean_brightness": round(self.mean_brightness, 1),
            "glare_percentage": round(self.glare_percentage, 2),
            "fov_fraction": round(self.fov_fraction, 3),
            "rb_ratio": round(self.rb_ratio, 2),
            "issues": self.reasons,
            "technician_actions": self.technician_actions,
        }


def check_image_gradability(
    img_bgr: np.ndarray,
    min_laplacian_var: float = 45.0,
    min_mean_brightness: float = 20.0,
    max_mean_brightness: float = 215.0,
    max_glare_percentage: float = 8.0,
    min_fov_fraction: float = 0.18,
    min_rb_ratio: float = 1.65,
) -> QualityReport:
    """
    Evaluates whether an image meets clinical standards for automated DR grading.
    Returns a comprehensive QualityReport with actionable technician instructions.
    """
    if img_bgr is None or img_bgr.size == 0:
        return QualityReport(
            is_gradable=False,
            quality_score=0.0,
            blur_score=0.0,
            mean_brightness=0.0,
            glare_percentage=100.0,
            fov_fraction=0.0,
            rb_ratio=0.0,
            reasons=["Empty or corrupted image data."],
            technician_actions=["Please verify camera connection and upload a valid image."],
        )

    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Blur Detection (Variance of Laplacian)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # 2. Illumination / Brightness
    mean_brightness = float(gray.mean())

    # 3. Specular Glare / Flash Artifacts (> 245 in all channels)
    glare_mask = (img_bgr[:, :, 0] > 245) & (img_bgr[:, :, 1] > 245) & (img_bgr[:, :, 2] > 245)
    glare_percentage = float((glare_mask.sum() / (h * w)) * 100.0)

    # 4. Retinal Field-of-View Masking
    _, thresh = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)
    active_pixels = int(thresh.sum() / 255)
    fov_fraction = float(active_pixels / (h * w))

    # 5. Chromaticity Check: Fundus Hemoglobin vs External Eye Photo
    b_chan = img_bgr[:, :, 0].astype(np.float32)
    g_chan = img_bgr[:, :, 1].astype(np.float32)
    r_chan = img_bgr[:, :, 2].astype(np.float32)
    lum_mask = (r_chan + g_chan + b_chan) > 35.0

    if lum_mask.sum() > 250:
        mean_r = float(r_chan[lum_mask].mean())
        mean_b = float(b_chan[lum_mask].mean())
        rb_ratio = float(mean_r / (mean_b + 1e-5))
    else:
        rb_ratio = 1.0

    reasons: List[str] = []
    actions: List[str] = []

    # Apply Quality Rules
    if blur_score < min_laplacian_var:
        reasons.append(f"Excessive motion or optical blur (Sharpness {blur_score:.1f} < threshold {min_laplacian_var:.1f}).")
        actions.append("Hold the fundus camera steady. Instruct patient to fixate on the internal fixation target.")

    if mean_brightness < min_mean_brightness:
        reasons.append(f"Image is severely underexposed (Mean brightness {mean_brightness:.1f} < {min_mean_brightness:.1f}).")
        actions.append("Increase camera flash/LED illumination intensity. Ensure the examination room is dimly lit.")
    elif mean_brightness > max_mean_brightness:
        reasons.append(f"Image is overexposed / washed out (Mean brightness {mean_brightness:.1f} > {max_mean_brightness:.1f}).")
        actions.append("Decrease camera flash energy to avoid bleaching delicate retinal microvasculature.")

    if glare_percentage > max_glare_percentage:
        reasons.append(f"Excessive corneal specular glare ({glare_percentage:.1f}% > {max_glare_percentage:.1f}% threshold).")
        actions.append("Slightly tilt or re-angle camera objective to disperse corneal light reflection away from the macula.")

    if fov_fraction < min_fov_fraction:
        reasons.append(f"Retinal field of view insufficient or off-center ({fov_fraction:.1%} < {min_fov_fraction:.1%}).")
        actions.append("Reposition patient chin rest and forehead bar. Center the camera objective directly over the pupil.")

    if rb_ratio < min_rb_ratio:
        reasons.append("Non-retinal or external ocular photo detected (cornea/sclera/face).")
        actions.append("Attach the 45-degree fundus condensing lens adapter to capture the interior posterior pole, not the external cornea.")

    # Calculate overall continuous quality score [0.0 - 1.0]
    blur_factor = min(1.0, blur_score / (min_laplacian_var * 2.0))
    exp_factor = 1.0 - min(1.0, abs(mean_brightness - 110.0) / 100.0)
    fov_factor = min(1.0, fov_fraction / 0.50)
    glare_penalty = max(0.0, 1.0 - (glare_percentage / 10.0))
    chroma_factor = 1.0 if rb_ratio >= min_rb_ratio else 0.2

    quality_score = float(np.clip(
        0.35 * blur_factor + 0.25 * exp_factor + 0.20 * fov_factor + 0.10 * glare_penalty + 0.10 * chroma_factor,
        0.0, 1.0
    ))

    is_gradable = (len(reasons) == 0)

    return QualityReport(
        is_gradable=is_gradable,
        quality_score=quality_score,
        blur_score=blur_score,
        mean_brightness=mean_brightness,
        glare_percentage=glare_percentage,
        fov_fraction=fov_fraction,
        rb_ratio=rb_ratio,
        reasons=reasons,
        technician_actions=actions,
        metrics={
            "blur_score": blur_score,
            "mean_brightness": mean_brightness,
            "glare_percentage": glare_percentage,
            "fov_fraction": fov_fraction,
            "rb_ratio": rb_ratio,
        }
    )
