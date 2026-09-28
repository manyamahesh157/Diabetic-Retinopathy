"""Unit tests for Clinical Quality Gate & Gradability Assessor."""

import numpy as np
import cv2
import pytest
from sanjeevani_ai.preprocessing.quality import check_image_gradability


def test_quality_gate_normal():
    # Synthetic normal fundus with red-dominant hemoglobin and good contrast
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (150, 150), 120, (30, 80, 200), -1)  # High Red, low Blue
    # Add high-frequency texture so Laplacian variance is high
    noise = np.random.randint(0, 50, (300, 300, 3), dtype=np.uint8)
    img = cv2.add(img, noise)

    report = check_image_gradability(img)
    assert report.is_gradable is True
    assert report.quality_score > 0.4
    assert len(report.reasons) == 0


def test_quality_gate_blur_rejection():
    # Heavily blurred image
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (150, 150), 120, (30, 80, 200), -1)
    blurred = cv2.GaussianBlur(img, (51, 51), 30)

    report = check_image_gradability(blurred, min_laplacian_var=45.0)
    assert report.is_gradable is False
    assert any("blur" in r.lower() for r in report.reasons)
    assert len(report.technician_actions) > 0


def test_quality_gate_underexposed():
    # Dark image
    img = np.ones((300, 300, 3), dtype=np.uint8) * 5
    report = check_image_gradability(img, min_mean_brightness=20.0)
    assert report.is_gradable is False
    assert any("underexposed" in r.lower() for r in report.reasons)


def test_quality_gate_overexposed():
    # Very bright washed-out image
    img = np.ones((300, 300, 3), dtype=np.uint8) * 230
    report = check_image_gradability(img, max_mean_brightness=215.0)
    assert report.is_gradable is False
    assert any("overexposed" in r.lower() for r in report.reasons)


def test_quality_gate_external_eye_rejection():
    # External ocular photo: white sclera / cornea has balanced RGB (R/B ratio ~ 1.0)
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (150, 150), 120, (180, 180, 190), -1)  # balanced RGB
    report = check_image_gradability(img, min_rb_ratio=1.65)
    assert report.is_gradable is False
    assert any("external" in r.lower() for r in report.reasons)
