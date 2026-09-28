"""Unit tests for Preprocessing Pipeline & Normalization."""

import numpy as np
import cv2
import torch
import pytest
from sanjeevani_ai.preprocessing.crop import crop_to_retinal_fov
from sanjeevani_ai.preprocessing.normalization import (
    apply_ben_graham_normalization,
    apply_green_channel_clahe,
    normalize_to_tensor,
)
from sanjeevani_ai.preprocessing.pipeline import PreprocessingPipeline


def test_crop_to_retinal_fov():
    # Frame of 400x400 with black border and centered circle of 200x200
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(frame, (200, 200), 100, (50, 100, 200), -1)

    cropped, mask, bbox = crop_to_retinal_fov(frame)
    x, y, w, h = bbox
    assert w > 0 and h > 0
    assert cropped.shape[0] < 400 or cropped.shape[1] < 400
    assert mask.shape[:2] == cropped.shape[:2]


def test_ben_graham_normalization():
    img_rgb = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)
    norm = apply_ben_graham_normalization(img_rgb, sigma=10)
    assert norm.shape == img_rgb.shape
    assert norm.dtype == np.uint8


def test_green_channel_clahe():
    img_rgb = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)
    clahe_img = apply_green_channel_clahe(img_rgb)
    assert clahe_img.shape == img_rgb.shape
    assert clahe_img.dtype == np.uint8


def test_preprocessing_pipeline_end_to_end():
    pipe = PreprocessingPipeline(target_size=256, normalization_method="ben_graham")
    # Synthetic fundus
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (150, 150), 120, (30, 80, 200), -1)
    noise = np.random.randint(0, 50, (300, 300, 3), dtype=np.uint8)
    img = cv2.add(img, noise)

    res = pipe.process(img)
    assert res["success"] is True
    assert res["tensor"].shape == (1, 3, 256, 256)
    assert res["normalized_float"].shape == (256, 256, 3)
