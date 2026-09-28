"""Unit tests for Model Components, Concept Bottleneck, and Severity Heads."""

import torch
import pytest
from sanjeevani_ai.models.cbam import CBAM
from sanjeevani_ai.models.backbone import RetinalBackbone
from sanjeevani_ai.models.concept_bottleneck import ConceptBottleneck, ConceptHead
from sanjeevani_ai.models.severity_head import SeverityGradingHead
from sanjeevani_ai.models.sanjeevani_model import SanjeevaniModel


def test_cbam_attention_forward():
    cbam = CBAM(planes=32)
    x = torch.randn(2, 32, 16, 16)
    out = cbam(x)
    assert out.shape == x.shape
    assert cbam.last_spatial_map is not None
    assert cbam.last_spatial_map.shape == (2, 1, 16, 16)


def test_concept_head_forward():
    head = ConceptHead(in_features=64)
    z = torch.randn(4, 64)
    p_logit, sev = head(z)
    assert p_logit.shape == (4,)
    assert sev.shape == (4,)
    assert (sev >= 0.0).all() and (sev <= 1.0).all()


def test_concept_bottleneck_forward():
    cbm = ConceptBottleneck(in_features=64)
    z = torch.randn(2, 64)
    out = cbm(z)
    assert "concept_vector" in out
    assert out["concept_vector"].shape == (2, 8)
    assert out["concept_probs"].shape == (2, 8)


def test_severity_head_forward():
    s_head = SeverityGradingHead(num_concepts=8, num_classes=5)
    c_vec = torch.rand(3, 8)
    logits, attributions = s_head(c_vec)
    assert logits.shape == (3, 5)
    assert attributions.shape == (3, 5, 8)


def test_full_sanjeevani_model_forward():
    model = SanjeevaniModel(backbone_name="efficientnet_b0", pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    assert out["logits"].shape == (1, 5)
    assert out["probs"].shape == (1, 5)
    assert out["concept_vector"].shape == (1, 8)
    assert torch.isclose(out["probs"].sum(), torch.tensor(1.0), atol=1e-5)
