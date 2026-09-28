"""Unit tests for Uncertainty Quantification (MC-Dropout and TTA)."""

import torch
import pytest
from sanjeevani_ai.models.sanjeevani_model import SanjeevaniModel
from sanjeevani_ai.models.uncertainty import (
    estimate_uncertainty_mc_dropout,
    predict_with_tta,
    quantify_prediction_uncertainty,
)


@pytest.fixture
def dummy_model():
    return SanjeevaniModel(backbone_name="efficientnet_b0", pretrained=False)


def test_mc_dropout_uncertainty(dummy_model):
    x = torch.randn(1, 3, 224, 224)
    mean_probs, entropy, mutual_info, max_var, history = estimate_uncertainty_mc_dropout(
        dummy_model, x, n_passes=5
    )
    assert mean_probs.shape == (5,)
    assert entropy >= 0.0
    assert mutual_info >= 0.0
    assert len(history) == 5


def test_predict_with_tta(dummy_model):
    x = torch.randn(1, 3, 224, 224)
    mean_probs, tta_var = predict_with_tta(dummy_model, x, n_variants=3)
    assert mean_probs.shape == (5,)
    assert tta_var >= 0.0


def test_quantify_prediction_uncertainty_levels(dummy_model):
    x = torch.randn(1, 3, 224, 224)
    fused_probs, report = quantify_prediction_uncertainty(
        dummy_model, x, n_mc_passes=3, n_tta_variants=2
    )
    assert report.uncertainty_level in ["LOW", "MEDIUM", "HIGH"]
    assert isinstance(report.is_flagged_for_review, bool)
    assert len(report.clinical_note) > 0
