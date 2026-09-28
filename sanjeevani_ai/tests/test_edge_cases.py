"""Unit tests for Edge Cases and Robust Failure Handling."""

import numpy as np
import pytest
from sanjeevani_ai.preprocessing.pipeline import PreprocessingPipeline
from sanjeevani_ai.inference.pipeline import SanjeevaniInferenceEngine


@pytest.fixture
def engine():
    return SanjeevaniInferenceEngine(force_demo_mode=True)


def test_edge_case_completely_black_image(engine):
    black_img = np.zeros((300, 300, 3), dtype=np.uint8)
    res = engine.predict(black_img)
    assert res["success"] is False
    assert res["is_gradable"] is False
    assert any("underexposed" in r.lower() for r in res["quality_report"].reasons)


def test_edge_case_all_white_overexposed(engine):
    white_img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    res = engine.predict(white_img)
    assert res["success"] is False
    assert res["is_gradable"] is False
    assert any("overexposed" in r.lower() or "glare" in r.lower() for r in res["quality_report"].reasons)


def test_edge_case_tiny_image(engine):
    # Very small image (12x12)
    tiny_img = np.random.randint(0, 255, (12, 12, 3), dtype=np.uint8)
    res = engine.predict(tiny_img)
    # Must fail or reject gracefully without crash
    assert "success" in res


def test_edge_case_corrupted_bytes(engine):
    corrupt_bytes = b"NOT_A_VALID_JPEG_IMAGE_DATA_HEADER"
    res = engine.predict(corrupt_bytes)
    assert res["success"] is False
    assert "error" in res


def test_edge_case_non_square_aspect_ratio(engine):
    # Retinal strip (150x600)
    strip_img = np.zeros((150, 600, 3), dtype=np.uint8)
    # Circle in center
    import cv2
    cv2.circle(strip_img, (300, 75), 65, (30, 80, 200), -1)
    noise = np.random.randint(0, 40, (150, 600, 3), dtype=np.uint8)
    strip_img = cv2.add(strip_img, noise)

    res = engine.predict(strip_img)
    assert "success" in res
