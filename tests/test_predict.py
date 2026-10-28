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
from custom_cnn import CustomCNN

sys.path.insert(0, str(Path(__file__).parent.parent))
import predict as pred_module
from predict import _model_cache, load_model, predict_image


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures — tiny dummy model
# ──────────────────────────────────────────────────────────────────────────────

NUM_CLASSES = 5
DUMMY_CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
]


class MockGate:
    def __init__(self, accepted: bool = True, score: float = 0.99):
        self.accepted = accepted
        self.score = score

    def accepts(self, image):
        return self.accepted, self.score


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
    result = predict_image(sample_image, model, classes, device, leaf_gate=MockGate())
    required = {"class_name", "crop", "disease", "confidence", "top_k", "treatment"}
    assert required.issubset(result.keys()), f"Missing keys: {required - result.keys()}"


def test_confidence_is_between_0_and_1(dummy_model_and_classes, sample_image):
    model, classes, device, _, _ = dummy_model_and_classes
    result = predict_image(sample_image, model, classes, device, leaf_gate=MockGate())
    assert 0.0 <= result["confidence"] <= 1.0


def test_top_k_length(dummy_model_and_classes, sample_image):
    model, classes, device, _, _ = dummy_model_and_classes
    result = predict_image(sample_image, model, classes, device, top_k=3, leaf_gate=MockGate())
    assert len(result["top_k"]) == 3


def test_top_k_confidences_sum_close_to_1(dummy_model_and_classes, sample_image):
    """Top-k confidences should not individually exceed 1."""
    model, classes, device, _, _ = dummy_model_and_classes
    result = predict_image(sample_image, model, classes, device, top_k=NUM_CLASSES, leaf_gate=MockGate())
    for entry in result["top_k"]:
        assert 0.0 <= entry["confidence"] <= 1.0


def test_non_image_file_raises_value_error(dummy_model_and_classes, bad_file):
    model, classes, device, _, _ = dummy_model_and_classes
    with pytest.raises(ValueError, match="Unsupported file type"):
        predict_image(bad_file, model, classes, device, leaf_gate=MockGate())


def test_class_name_in_classes_list(dummy_model_and_classes, sample_image):
    model, classes, device, _, _ = dummy_model_and_classes
    result = predict_image(sample_image, model, classes, device, leaf_gate=MockGate())
    assert result["class_name"] in classes


def test_below_threshold_returns_no_disease_or_treatment(dummy_model_and_classes, sample_image):
    model, classes, device, _, _ = dummy_model_and_classes
    result = predict_image(sample_image, model, classes, device, leaf_gate=MockGate(False, 0.1))
    assert result["status"] == "not_a_leaf"
    assert result["leaf_score"] == 0.1
    assert "disease" not in result
    assert "treatment" not in result


def test_custom_backend_loads_and_predicts(tmp_path, sample_image):
    classes_path = tmp_path / "classes.json"
    classes_path.write_text(json.dumps(DUMMY_CLASSES), encoding="utf-8")
    checkpoint_path = tmp_path / "custom_cnn.pt"
    custom = CustomCNN(NUM_CLASSES, input_size=128)
    torch.save({"state_dict": custom.state_dict(), "input_size": 128, "classes": DUMMY_CLASSES}, checkpoint_path)
    _model_cache.clear()
    model, classes, device = load_model(checkpoint_path, classes_path, torch.device("cpu"), backend="custom")
    result = predict_image(sample_image, model, classes, device, leaf_gate=MockGate())
    assert result["status"] == "ok"
    assert result["class_name"] in DUMMY_CLASSES


def test_prediction_masks_classes_not_trained_by_custom_checkpoint(sample_image):
    class FixedLogitModel(torch.nn.Module):
        input_size = 224
        active_class_indices = [1, 3]

        def forward(self, tensor):
            # Index 0 would win without the custom-checkpoint class mask.
            return torch.tensor([[10.0, 2.0, 3.0, 4.0]], device=tensor.device)

    classes = [
        "Tomato_healthy",
        "Tomato_Late_blight",
        "Tomato_Early_blight",
        "Tomato_Leaf_Mold",
    ]
    result = predict_image(
        sample_image,
        FixedLogitModel(),
        classes,
        torch.device("cpu"),
        top_k=3,
        leaf_gate=MockGate(),
    )

    assert result["class_name"] == "Tomato_Leaf_Mold"
    assert {entry["class_name"] for entry in result["top_k"]} == {
        "Tomato_Late_blight",
        "Tomato_Leaf_Mold",
    }
