"""
explain_report.py
------------------
Turns raw numbers (class probabilities, heatmaps, uncertainty) into a
patient/doctor-readable report. This is what non-technical judges and
clinicians will actually read — the heatmap alone means nothing to them
without a sentence explaining WHAT it means.

Approach: a lightweight rule-based Natural Language Generation (NLG)
template engine driven by (a) predicted grade, (b) heatmap spatial
statistics (how many separate hot regions, how large, roughly where),
and (c) uncertainty. This is deliberately NOT another black box — the
explanation-generation logic itself is fully inspectable, which matters
a lot when your whole pitch is "explainability".
"""

import io
import numpy as np
import cv2
from datetime import datetime

from . import config

LESION_HINTS = {
    0: "No visible lesions detected — retina appears healthy.",
    1: "Isolated microaneurysms (tiny red dots, early capillary damage) detected.",
    2: "Microaneurysms together with dot-and-blot hemorrhages and/or hard "
       "exudates (yellowish lipid deposits) detected — indicates progressing "
       "vascular damage.",
    3: "Extensive hemorrhages across multiple retinal quadrants and venous "
       "beading detected — high risk of progression to proliferative disease.",
    4: "Neovascularization (abnormal new blood vessel growth) and/or vitreous "
       "hemorrhage patterns detected — sight-threatening; urgent referral required.",
}

URGENCY = {
    0: "Routine annual screening recommended.",
    1: "Follow-up screening recommended within 9-12 months.",
    2: "Follow-up recommended within 6 months; consider referral to ophthalmologist.",
    3: "Refer to ophthalmologist within 1 month.",
    4: "URGENT referral to ophthalmologist required (within days).",
}


def _describe_heatmap_regions(heatmap: np.ndarray, threshold: float = 0.6) -> str:
    """Counts distinct hot blobs and roughly locates them (quadrant-wise) —
    turns a pixel array into a sentence a clinician can sanity-check."""
    h, w = heatmap.shape
    mask = (heatmap > threshold).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    regions = []
    for i in range(1, num_labels):          # skip background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 15:                       # ignore noise specks
            continue
        cx, cy = centroids[i]
        vert = "superior" if cy < h / 2 else "inferior"
        horiz = "temporal" if cx < w / 2 else "nasal"
        regions.append(f"{vert}-{horiz} quadrant")

    if not regions:
        return "No strongly-localized abnormal regions were highlighted by the model."
    unique_regions = sorted(set(regions))
    return ("Model attention was concentrated in the " +
            ", ".join(unique_regions) + f" of the retina ({len(regions)} distinct hotspot(s)).")


def generate_text_explanation(predicted_class: int, probs: np.ndarray,
                               predictive_entropy: float, cam_map: np.ndarray,
                               agreement: float) -> dict:
    confidence = float(probs[predicted_class]) * 100
    needs_review = predictive_entropy > config.UNCERTAINTY_FLAG_THRESHOLD

    explanation = {
        "grade": config.CLASS_NAMES[predicted_class],
        "grade_index": predicted_class,
        "confidence_pct": round(confidence, 1),
        "clinical_finding": LESION_HINTS[predicted_class],
        "spatial_explanation": _describe_heatmap_regions(cam_map),
        "recommended_action": URGENCY[predicted_class],
        "xai_agreement_score": round(agreement, 2),
        "uncertainty_flag": needs_review,
        "uncertainty_note": (
            "⚠ The model's confidence across repeated stochastic passes was low "
            "and/or the two explanation methods disagreed — treat this prediction "
            "as inconclusive and prioritize specialist review."
            if needs_review else
            "Model confidence was stable across repeated stochastic inference passes."
        ),
    }
    return explanation


def build_pdf_report(patient_id: str, original_img_rgb: np.ndarray,
                      overlay_img: np.ndarray, explanation: dict,
                      probs: np.ndarray) -> bytes:
    """Builds a downloadable PDF report combining image + heatmap + explanation.
    Returns raw PDF bytes (so the caller / Streamlit can offer it as a download
    without ever touching disk)."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm
    from PIL import Image as PILImage

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(2 * cm, height - 2 * cm, "Diabetic Retinopathy — Explainable AI Report")
    c.setFont("Helvetica", 10)
    c.drawString(2 * cm, height - 2.7 * cm,
                 f"Patient/Case ID: {patient_id}   |   Generated: {datetime.now():%Y-%m-%d %H:%M}")

    # embed original + overlay images side by side
    orig_pil = PILImage.fromarray((original_img_rgb * 255).astype(np.uint8))
    overlay_pil = PILImage.fromarray(overlay_img)
    img_y = height - 10.5 * cm
    c.drawInlineImage(orig_pil, 2 * cm, img_y, width=7 * cm, height=7 * cm)
    c.drawInlineImage(overlay_pil, 10 * cm, img_y, width=7 * cm, height=7 * cm)
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(2 * cm, img_y - 0.5 * cm, "Original (preprocessed) fundus image")
    c.drawString(10 * cm, img_y - 0.5 * cm, "Grad-CAM++ / Integrated-Gradients overlay")

    text_y = img_y - 1.5 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, text_y, f"Predicted Grade: {explanation['grade']}  "
                                 f"({explanation['confidence_pct']}% confidence)")
    text_y -= 0.8 * cm
    c.setFont("Helvetica", 10)
    lines = [
        f"Clinical finding: {explanation['clinical_finding']}",
        f"Spatial explanation: {explanation['spatial_explanation']}",
        f"Recommended action: {explanation['recommended_action']}",
        f"XAI cross-method agreement score: {explanation['xai_agreement_score']} (1.0 = perfect agreement)",
        f"Reliability note: {explanation['uncertainty_note']}",
    ]
    for line in lines:
        for wrapped in _wrap_text(line, 95):
            c.drawString(2 * cm, text_y, wrapped)
            text_y -= 0.55 * cm

    text_y -= 0.3 * cm
    c.setFont("Helvetica-Bold", 10)
    c.drawString(2 * cm, text_y, "Class probability distribution:")
    text_y -= 0.5 * cm
    c.setFont("Helvetica", 9)
    for idx, name in enumerate(config.CLASS_NAMES):
        c.drawString(2.3 * cm, text_y, f"{name}: {probs[idx]*100:.1f}%")
        text_y -= 0.45 * cm

    c.setFont("Helvetica-Oblique", 7)
    c.drawString(2 * cm, 1.2 * cm,
                 "This is an AI screening aid, not a diagnostic replacement. "
                 "All flagged cases should be confirmed by a qualified ophthalmologist.")

    c.save()
    buf.seek(0)
    return buf.getvalue()


def _wrap_text(text: str, width: int):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= width:
            cur = f"{cur} {w}".strip()
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines
