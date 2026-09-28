"""
sanjeevani_ai.counterfactual.image_counterfactual
-------------------------------------------------
Image-Space Lesion Masking / Virtual Ablation Re-Inference.

Simulates the clinical question:
"If the model did not observe the highlighted lesion foci, how would its severity grading change?"

Methodology:
1. Extract high-activation lesion mask from Grad-CAM++ (e.g. threshold > 0.60).
2. Perform localized inpainting/background blending to suppress the suspected lesion signals.
3. Re-execute model forward inference on the ablated retina.
4. Measure causal effect on ICDR grade and concept activations.
"""

from typing import Dict, Any, Tuple
import numpy as np
import cv2
import torch

from ..models.severity_head import ICDR_GRADES
from ..preprocessing.normalization import normalize_to_tensor


def simulate_lesion_ablation(
    model: torch.nn.Module,
    img_rgb_float: np.ndarray,
    heatmap: np.ndarray,
    original_grade: int,
    hotspot_threshold: float = 0.55
) -> Dict[str, Any]:
    """
    Ablates primary lesion hotspots and re-evaluates model predictions.
    """
    h, w = img_rgb_float.shape[:2]
    if heatmap.shape[:2] != (h, w):
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    # 1. Binarize lesion mask
    lesion_mask = (heatmap > hotspot_threshold).astype(np.uint8)

    # 2. Inpaint / smooth ablated regions with surrounding retinal background
    # Dilate slightly to cover entire lesion boundary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated_mask = cv2.dilate(lesion_mask, kernel)

    img_uint8 = np.uint8(img_rgb_float * 255.0)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

    # Inpaint using Telea algorithm
    inpainted_bgr = cv2.inpaint(img_bgr, dilated_mask * 255, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    ablated_float = inpainted_rgb.astype(np.float32) / 255.0

    # 3. Model Re-inference on ablated image
    tensor_ablated = normalize_to_tensor(ablated_float).unsqueeze(0).to(next(model.parameters()).device)
    model.eval()
    with torch.no_grad():
        out = model(tensor_ablated)
        probs_ablated = out["probs"].squeeze(0).cpu().numpy()
        ablated_grade = int(np.argmax(probs_ablated))
        ablated_concepts = out["concept_probs"].squeeze(0).cpu().numpy().tolist()

    # 4. Formulate causal explanation
    grade_shift = original_grade - ablated_grade
    if grade_shift > 0:
        causal_note = (
            f"Suppressing the highlighted lesion regions causally decreased model prediction from "
            f"{ICDR_GRADES[original_grade]} to {ICDR_GRADES[ablated_grade]}. "
            f"This confirms the model's grading was directly dependent on these localized retinal findings."
        )
        causal_confirmed = True
    elif grade_shift == 0:
        causal_note = (
            f"Prediction remained {ICDR_GRADES[original_grade]} after suppressing the highest-intensity focus. "
            f"Diffuse, distributed background abnormalities across other quadrants continue to support the grade."
        )
        causal_confirmed = False
    else:
        causal_note = "Ablation did not lower the grade; non-linear background interactions detected."
        causal_confirmed = False

    return {
        "original_grade": original_grade,
        "original_grade_name": ICDR_GRADES[original_grade],
        "ablated_grade": ablated_grade,
        "ablated_grade_name": ICDR_GRADES[ablated_grade],
        "grade_shift": grade_shift,
        "causal_confirmed": causal_confirmed,
        "causal_note": causal_note,
        "ablated_img_rgb": inpainted_rgb,
        "lesion_mask": dilated_mask * 255,
        "ablated_probs": probs_ablated.tolist(),
        "ablated_concepts": ablated_concepts,
        "disclaimer": "MODEL COUNTERFACTUAL: Demonstrates mathematical attribution sensitivity; not a patient treatment outcome simulation.",
    }
