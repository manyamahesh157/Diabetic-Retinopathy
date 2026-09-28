"""
sanjeevani_ai.xai.concept_explanations
--------------------------------------
Concept Bottleneck Decomposition & Anatomical Retinal Quadrant Profiling.

Transforms intermediate concept vectors into structured clinical evidence:
- Lesion presence & confidence
- Concept-to-grade causal weight attribution
- Retinal quadrant burden (Superior-Temporal, Superior-Nasal, Inferior-Temporal, Inferior-Nasal)
- Macular / CSME involvement risk
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
import torch

from ..models.concept_bottleneck import CLINICAL_CONCEPTS


def analyze_retinal_quadrants(heatmap: np.ndarray, threshold: float = 0.55) -> Dict[str, Any]:
    """
    Partitions the retinal field into 4 clinical quadrants around the optical center:
    - Superior-Temporal (ST)
    - Superior-Nasal (SN)
    - Inferior-Temporal (IT)
    - Inferior-Nasal (IN)
    Returns hotspot counts, quadrant burden percentages, and connected component centroids.
    """
    h, w = heatmap.shape[:2]
    mid_y, mid_x = h // 2, w // 2

    active_mask = (heatmap > threshold).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(active_mask, connectivity=8)

    quadrant_counts = {
        "Superior-Temporal": 0,
        "Superior-Nasal": 0,
        "Inferior-Temporal": 0,
        "Inferior-Nasal": 0,
    }

    focal_regions = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 12:  # Filter out minor noise specks
            continue
        cx, cy = centroids[i]
        vert = "Superior" if cy < mid_y else "Inferior"
        # In fundus photography convention: temporal is usually lateral to macula
        horiz = "Temporal" if cx < mid_x else "Nasal"
        q_name = f"{vert}-{horiz}"
        quadrant_counts[q_name] += 1
        focal_regions.append({
            "quadrant": q_name,
            "center": (int(cx), int(cy)),
            "area": int(area),
            "peak_intensity": float(heatmap[labels == i].max())
        })

    # Macular zone check (within 1/6 radius of center)
    center_y, center_x = mid_y, mid_x
    macular_radius = min(h, w) // 6
    y_coords, x_coords = np.ogrid[:h, :w]
    macular_mask = ((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2) <= (macular_radius ** 2)
    macular_intensity = float(heatmap[macular_mask].mean()) if macular_mask.any() else 0.0

    return {
        "quadrant_counts": quadrant_counts,
        "total_hotspots": len(focal_regions),
        "focal_regions": focal_regions,
        "macular_zone_burden": macular_intensity,
        "macular_threat": macular_intensity > 0.40,
    }


def format_concept_evidence(
    concept_probs: np.ndarray,
    concept_severities: np.ndarray,
    concept_attributions: np.ndarray,
    predicted_grade: int,
    concept_specs: List[Dict[str, Any]] = CLINICAL_CONCEPTS
) -> List[Dict[str, Any]]:
    """
    Assembles comprehensive clinical concept evidence breakdown for report and UI.
    """
    evidence = []
    # If 2D arrays, take first batch item
    if concept_probs.ndim > 1:
        concept_probs = concept_probs[0]
    if concept_severities.ndim > 1:
        concept_severities = concept_severities[0]
    if concept_attributions.ndim > 2:
        concept_attributions = concept_attributions[0, predicted_grade]
    elif concept_attributions.ndim == 2:
        concept_attributions = concept_attributions[predicted_grade]

    for idx, spec in enumerate(concept_specs):
        p = float(concept_probs[idx])
        sev = float(concept_severities[idx])
        attr = float(concept_attributions[idx])

        # Clinical evidence classification
        if p >= 0.70:
            status_text = "Detected (Strong Evidence)"
            badge = "DETECTED"
        elif p >= 0.40:
            status_text = "Moderately Detected"
            badge = "MODERATE"
        elif p >= 0.20:
            status_text = "Low Evidence / Inconclusive"
            badge = "LOW"
        else:
            status_text = "Not Confidently Detected"
            badge = "ABSENT"

        evidence.append({
            "id": spec["id"],
            "name": spec["name"],
            "category": spec["category"],
            "validation_status": spec["status"],
            "dataset_source": spec["source"],
            "presence_prob": round(p, 3),
            "severity_score": round(sev, 3),
            "attribution_to_grade": round(attr, 3),
            "status_text": status_text,
            "badge": badge,
            "supports_grade": (attr > 0.05),
        })

    return evidence
