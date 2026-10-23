"""
tests/test_predict.py
=====================
Tests for the predict.py prediction pipeline.
Uses a tiny randomly-initialised EfficientNet-B0 (no real weights needed).
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision import models

sys.path.insert(0, str(Path(__file__).parent.parent))
import predict as pred_module
from predict import predict_image


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures — tiny dummy model
# ──────────────────────────────────────────────────────────────────────────────

NUM_CLASSES = 5
DUMMY_CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Potato___Early_blight",
    "Potato___healthy",
]


@pytest.fixture(scope="module")
def dummy_model_and_classes(tmp_path_factory):
    """Build a tiny untrained EfficientNet-B0 and save it to disk."""
    tmp = tmp_path_factory.mktemp("model")
    model_path  = tmp / "best.pt"
    classes_path = tmp / "classes.json"

    model = models.efficientnet_b0(weights=None)
    in_feat = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_feat, NUM_CLASSES)
    torch.save(model.state_dict(), model_path)

    with open(classes_path, "w") as f:
        json.dump(DUMMY_CLASSES, f)

    return model, DUMMY_CLASSES, torch.device("cpu"), model_path, classes_path


@pytest.fixture()
def sample_image(tmp_path: Path) -> Path:
    """Create a small solid-colour PNG image."""
    img = Image.new("RGB", (256, 256), color=(100, 180, 60))
    img_path = tmp_path / "leaf.png"
    img.save(img_path)
    return img_path


@pytest.fixture()
def bad_file(tmp_path: Path) -> Path:
    """Create a non-image .txt file."""
    p = tmp_path / "bad.txt"
    p.write_text("This is not an image.")
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_predict_returns_required_keys(dummy_model_and_classes, sample_image):
    model, classes, device, _, _ = dummy_model_and_classes
    result = predict_image(sample_image, model, classes, device)
    required = {"class_name", "crop", "disease", "confidence", "top_k", "treatment"}
    assert required.issubset(result.keys()), f"Missing keys: {required - result.keys()}"


def test_confidence_is_between_0_and_1(dummy_model_and_classes, sample_image):
    model, classes, device, _, _ = dummy_model_and_classes
    result = predict_image(sample_image, model, classes, device)
    assert 0.0 <= result["confidence"] <= 1.0


def test_top_k_length(dummy_model_and_classes, sample_image):
    model, classes, device, _, _ = dummy_model_and_classes
    result = predict_image(sample_image, model, classes, device, top_k=3)
    assert len(result["top_k"]) == 3


def test_top_k_confidences_sum_close_to_1(dummy_model_and_classes, sample_image):
    """Top-k confidences should not individually exceed 1."""
    model, classes, device, _, _ = dummy_model_and_classes
    result = predict_image(sample_image, model, classes, device, top_k=NUM_CLASSES)
    for entry in result["top_k"]:
        assert 0.0 <= entry["confidence"] <= 1.0


def test_non_image_file_raises_value_error(dummy_model_and_classes, bad_file):
    model, classes, device, _, _ = dummy_model_and_classes
    with pytest.raises(ValueError, match="Unsupported file type"):
        predict_image(bad_file, model, classes, device)


def test_class_name_in_classes_list(dummy_model_and_classes, sample_image):
    model, classes, device, _, _ = dummy_model_and_classes
    result = predict_image(sample_image, model, classes, device)
    assert result["class_name"] in classes

