"""
infer.py
--------
The single function the app calls: raw camera/upload image bytes IN,
full explainable prediction OUT. This is the "one function that runs the
whole pipeline" you can point to during the demo/code-walkthrough.

Pipeline stages:
  1. Decode image
  2. Quality gate                (quality_check.py)
  3. Preprocess                  (preprocessing.py)
  4. Predict w/ uncertainty      (model.py: MC-Dropout + TTA)
  5. Explain (Grad-CAM++ + IG)   (xai_methods.py)
  6. Cross-check explanations -> agreement score
  7. Generate text + PDF report  (explain_report.py)
"""

import numpy as np
import torch
import cv2

from . import config
from .quality_check import check_image_quality
from .preprocessing import preprocess_fundus_image, to_model_tensor
from .model import DRGradingModel, predict_with_uncertainty, predict_with_tta
from .xai_methods import GradCAMPlusPlus, integrated_gradients, overlay_heatmap_on_image, agreement_score
from .explain_report import generate_text_explanation, build_pdf_report


_model_cache = {}


def load_model(checkpoint_path: str = config.CHECKPOINT_PATH) -> DRGradingModel:
    if "model" in _model_cache:
        return _model_cache["model"]
    model = DRGradingModel(pretrained=True).to(config.DEVICE)
    try:
        state = torch.load(checkpoint_path, map_location=config.DEVICE)
        model.load_state_dict(state)
    except FileNotFoundError:
        # No trained checkpoint yet — the ImageNet-pretrained backbone still
        # runs end-to-end for demo/plumbing purposes; swap in a fine-tuned
        # checkpoint (see train.py) before the real demo for real accuracy.
        pass
    model.eval()
    _model_cache["model"] = model
    return model


def run_full_pipeline(image_bytes: bytes, patient_id: str = "demo-case") -> dict:
    """Returns a dict with every artifact the UI needs: quality report,
    prediction, uncertainty, heatmaps, text explanation, and a ready PDF."""
    file_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return {"error": "Could not decode image. Please upload a valid JPG/PNG fundus photo."}

    quality = check_image_quality(img_bgr)
    if not quality.is_gradable:
        return {
            "error": "Image failed quality checks — please recapture.",
            "quality_reasons": quality.reasons,
        }

    img_rgb_float = preprocess_fundus_image(img_bgr, config.IMG_SIZE)
    tensor = to_model_tensor(img_rgb_float).unsqueeze(0).to(config.DEVICE)

    model = load_model()

    # Uncertainty-aware prediction (MC-Dropout) then refine with TTA
    mean_probs, pred_entropy, mutual_info = predict_with_uncertainty(model, tensor)
    tta_probs = predict_with_tta(model, tensor)
    final_probs = ((mean_probs + tta_probs) / 2).squeeze(0).detach().cpu().numpy()
    predicted_class = int(np.argmax(final_probs))
    entropy_val = float(pred_entropy.squeeze(0).item())

    # Explainability: Grad-CAM++ on the CBAM-attended feature map (multi-channel,
    # right before global pooling) — NOT the single-channel spatial-attention
    # mask, which would collapse the channel-weighting Grad-CAM++ relies on.
    target_layer = model.attention
    cam_gen = GradCAMPlusPlus(model, target_layer)
    tensor_grad = tensor.clone().requires_grad_(True)
    cam_map = cam_gen(tensor_grad, predicted_class)

    ig_map = integrated_gradients(model, tensor, predicted_class, steps=30)

    agree = agreement_score(cam_map, ig_map)
    overlay = overlay_heatmap_on_image(img_rgb_float, cam_map)

    explanation = generate_text_explanation(
        predicted_class, final_probs, entropy_val, cam_map, agree
    )

    pdf_bytes = build_pdf_report(patient_id, img_rgb_float, overlay, explanation, final_probs)

    return {
        "error": None,
        "quality": quality,
        "predicted_class": predicted_class,
        "class_name": config.CLASS_NAMES[predicted_class],
        "probs": final_probs,
        "predictive_entropy": entropy_val,
        "mutual_information": float(mutual_info.squeeze(0).item()),
        "cam_map": cam_map,
        "ig_map": ig_map,
        "overlay_image": overlay,
        "preprocessed_image": img_rgb_float,
        "xai_agreement": agree,
        "explanation": explanation,
        "pdf_bytes": pdf_bytes,
    }
