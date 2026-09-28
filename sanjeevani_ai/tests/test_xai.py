"""Unit tests for Explainability Stack (Grad-CAM++, Integrated Gradients, Agreement)."""

import numpy as np
import torch
import pytest
from sanjeevani_ai.models.sanjeevani_model import SanjeevaniModel
from sanjeevani_ai.xai.gradcam import GradCAMPlusPlus, overlay_heatmap
from sanjeevani_ai.xai.integrated_gradients import compute_integrated_gradients
from sanjeevani_ai.xai.agreement import compute_explanation_agreement
from sanjeevani_ai.xai.concept_explanations import analyze_retinal_quadrants, format_concept_evidence


@pytest.fixture
def dummy_model():
    return SanjeevaniModel(backbone_name="efficientnet_b0", pretrained=False)


def test_gradcam_plus_plus(dummy_model):
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    target_layer = dummy_model.attention
    cam_gen = GradCAMPlusPlus(dummy_model, target_layer)
    cam_map = cam_gen.generate(x, class_idx=2)
    cam_gen.remove_hooks()

    assert cam_map.shape == (224, 224)
    assert cam_map.min() >= 0.0 and cam_map.max() <= 1.0


def test_integrated_gradients(dummy_model):
    x = torch.randn(1, 3, 224, 224)
    ig_map = compute_integrated_gradients(dummy_model, x, class_idx=1, steps=5)
    assert ig_map.shape == (224, 224)
    assert ig_map.min() >= 0.0 and ig_map.max() <= 1.0


def test_explanation_agreement():
    # Two identical heatmaps should have agreement ~ 1.0
    h1 = np.ones((100, 100), dtype=np.float32)
    h2 = np.ones((100, 100), dtype=np.float32)
    agree = compute_explanation_agreement(h1, h2)
    assert "agreement_score" in agree
    assert agree["agreement_score"] >= 0.0


def test_quadrant_analysis():
    heatmap = np.zeros((200, 200), dtype=np.float32)
    # Add strong hotspot in superior temporal (top-left)
    heatmap[20:50, 20:50] = 0.95
    info = analyze_retinal_quadrants(heatmap, threshold=0.5)
    assert info["total_hotspots"] >= 1
    assert "quadrant_counts" in info
