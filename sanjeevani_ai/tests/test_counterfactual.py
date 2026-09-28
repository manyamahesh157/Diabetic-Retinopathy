"""Unit tests for Counterfactual Reasoning Engines."""

import numpy as np
import torch
import pytest
from sanjeevani_ai.models.severity_head import SeverityGradingHead
from sanjeevani_ai.models.sanjeevani_model import SanjeevaniModel
from sanjeevani_ai.counterfactual.concept_counterfactual import (
    optimize_concept_counterfactual,
    compute_borderline_counterfactuals,
)
from sanjeevani_ai.counterfactual.image_counterfactual import simulate_lesion_ablation


def test_concept_counterfactual_optimization():
    s_head = SeverityGradingHead(num_concepts=8, num_classes=5)
    # Moderate NPDR profile
    c_vector = torch.tensor([[0.8, 0.7, 0.6, 0.2, 0.0, 0.0, 0.0, 0.3]])

    # Target shift: Moderate (2) -> Mild (1)
    res = optimize_concept_counterfactual(
        s_head, c_vector, current_grade=2, target_grade=1, max_steps=40
    )
    assert "success" in res
    assert "concept_changes" in res
    assert len(res["summary"]) > 0


def test_borderline_counterfactuals():
    s_head = SeverityGradingHead(num_concepts=8, num_classes=5)
    c_vector = torch.tensor([[0.8, 0.7, 0.6, 0.2, 0.0, 0.0, 0.0, 0.3]])

    res = compute_borderline_counterfactuals(s_head, c_vector, predicted_grade=2)
    assert res["downward_transition"] is not None
    assert res["upward_transition"] is not None


def test_image_lesion_ablation():
    model = SanjeevaniModel(backbone_name="efficientnet_b0", pretrained=False)
    img_rgb_float = np.random.rand(224, 224, 3).astype(np.float32)
    heatmap = np.zeros((224, 224), dtype=np.float32)
    heatmap[80:120, 80:120] = 0.90  # Concentrated hotspot

    ablation_res = simulate_lesion_ablation(
        model, img_rgb_float, heatmap, original_grade=2, hotspot_threshold=0.5
    )
    assert "ablated_grade" in ablation_res
    assert "ablated_img_rgb" in ablation_res
    assert ablation_res["ablated_img_rgb"].shape == (224, 224, 3)
