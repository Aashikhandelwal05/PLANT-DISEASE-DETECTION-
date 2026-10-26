"""
tests/test_api.py
=================
Tests for the FastAPI endpoints in api.py.
Uses a tiny dummy model so no real weights are required.
Covers: 401 (bad API key), 415 (non-image upload), 422 (missing file),
        200 (valid image prediction).
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from PIL import Image
from fastapi.testclient import TestClient
from torchvision import models

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Dummy model helpers ───────────────────────────────────────────────────────

NUM_CLASSES = 5
DUMMY_CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Potato___Early_blight",
    "Potato___healthy",
]


class MockGate:
    def __init__(self, accepted: bool = True, score: float = 0.99):
        self.accepted = accepted
        self.score = score

    def accepts(self, image):
        return self.accepted, self.score


def _build_dummy_model():
    m = models.efficientnet_b0(weights=None)
    in_feat = m.classifier[1].in_features
    m.classifier[1] = torch.nn.Linear(in_feat, NUM_CLASSES)
    return m.eval()


def _green_leaf_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), color=(80, 170, 60)).save(buf, format="PNG")
    return buf.getvalue()


# ── Patch API state before importing api module ───────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def _patch_api_state():
    """Replace the global model/classes/device in api.py with dummy values."""
    import api
    dummy = _build_dummy_model()
    api._model   = dummy
    api._classes = DUMMY_CLASSES
    api._device  = torch.device("cpu")
    api._custom_model = dummy
    api._custom_classes = DUMMY_CLASSES
    api._custom_device = torch.device("cpu")
    api._API_KEY = "test-secret-key"
    api._leaf_gate = MockGate()
    yield
    # Restore
    api._model   = None
    api._classes = []
    api._custom_model = None
    api._custom_classes = []
    api._custom_device = None
    api._API_KEY = ""
    api._leaf_gate = None


@pytest.fixture(scope="module")
def client():
    import api
    return TestClient(api.app, raise_server_exceptions=False)


VALID_HEADERS = {"X-API-Key": "test-secret-key"}

# ──────────────────────────────────────────────────────────────────────────────
# Health endpoint
# ──────────────────────────────────────────────────────────────────────────────

def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_ready"] is True


# ──────────────────────────────────────────────────────────────────────────────
# Authentication
# ──────────────────────────────────────────────────────────────────────────────

def test_missing_api_key_returns_401(client):
    img_bytes = _green_leaf_bytes()
    resp = client.post(
        "/api/v1/predict",
        files={"file": ("leaf.png", img_bytes, "image/png")},
    )
    assert resp.status_code == 401


def test_wrong_api_key_returns_401(client):
    img_bytes = _green_leaf_bytes()
    resp = client.post(
        "/api/v1/predict",
        headers={"X-API-Key": "wrong-key"},
        files={"file": ("leaf.png", img_bytes, "image/png")},
    )
    assert resp.status_code == 401


def test_not_a_leaf_has_no_disease_or_treatment(client):
    import api
    api._leaf_gate = MockGate(False, 0.1)
    resp = client.post(
        "/api/v1/predict",
        headers=VALID_HEADERS,
        files={"file": ("not-leaf.png", _green_leaf_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "not_a_leaf"
    assert body["leaf_score"] == 0.1
    assert "disease" not in body
    assert "treatment" not in body
    api._leaf_gate = MockGate()


def test_custom_endpoint_rejects_non_leaf_before_disease_inference(client):
    """The custom route must use the gate and must not need the default model."""
    import api
    api._model = None
    api._leaf_gate = MockGate(False, 0.1)
    resp = client.post(
        "/api/v1/predict/custom",
        headers=VALID_HEADERS,
        files={"file": ("not-leaf.png", _green_leaf_bytes(), "image/png")},
    )
    body = resp.json()
    assert resp.status_code == 200
    assert body["status"] == "not_a_leaf"
    assert body["leaf_score"] == 0.1
    assert "disease" not in body
    assert "treatment" not in body
    api._model = api._custom_model
    api._leaf_gate = MockGate()


def test_custom_endpoint_has_the_standard_success_response(client):
    resp = client.post(
        "/api/v1/predict/custom",
        headers=VALID_HEADERS,
        files={"file": ("leaf.png", _green_leaf_bytes(), "image/png")},
    )
    body = resp.json()
    required = {"status", "leaf_score", "crop", "disease", "class_name", "confidence", "top_3", "treatment"}
    assert resp.status_code == 200
    assert body["status"] == "ok"
    assert required.issubset(body.keys())


# ──────────────────────────────────────────────────────────────────────────────
# Unsupported media type
# ──────────────────────────────────────────────────────────────────────────────

def test_non_image_returns_415(client):
    txt_bytes = b"This is plain text, not an image."
    resp = client.post(
        "/api/v1/predict",
        headers=VALID_HEADERS,
        files={"file": ("readme.txt", txt_bytes, "text/plain")},
    )
    assert resp.status_code == 415


# ──────────────────────────────────────────────────────────────────────────────
# Missing file (422 Unprocessable Entity)
# ──────────────────────────────────────────────────────────────────────────────

def test_missing_file_returns_422(client):
    resp = client.post("/api/v1/predict", headers=VALID_HEADERS)
    assert resp.status_code == 422


# ──────────────────────────────────────────────────────────────────────────────
# Valid prediction
# ──────────────────────────────────────────────────────────────────────────────

def test_valid_image_returns_200(client):
    img_bytes = _green_leaf_bytes()
    resp = client.post(
        "/api/v1/predict",
        headers=VALID_HEADERS,
        files={"file": ("leaf.png", img_bytes, "image/png")},
    )
    assert resp.status_code == 200


def test_valid_image_response_has_required_keys(client):
    img_bytes = _green_leaf_bytes()
    resp = client.post(
        "/api/v1/predict",
        headers=VALID_HEADERS,
        files={"file": ("leaf.png", img_bytes, "image/png")},
    )
    body = resp.json()
    required = {"crop", "disease", "class_name", "confidence", "top_3", "treatment"}
    assert required.issubset(body.keys()), f"Missing keys: {required - body.keys()}"


def test_confidence_within_range(client):
    img_bytes = _green_leaf_bytes()
    resp = client.post(
        "/api/v1/predict",
        headers=VALID_HEADERS,
        files={"file": ("leaf.png", img_bytes, "image/png")},
    )
    body = resp.json()
    assert 0.0 <= body["confidence"] <= 1.0


def test_top3_has_three_entries(client):
    img_bytes = _green_leaf_bytes()
    resp = client.post(
        "/api/v1/predict",
        headers=VALID_HEADERS,
        files={"file": ("leaf.png", img_bytes, "image/png")},
    )
    body = resp.json()
    assert len(body["top_3"]) == 3
