"""
tests/test_treatments.py
========================
Tests that every class in classes.json has an entry in treatments.json.
Uses a real treatments.json but a small dummy classes.json for quick runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

TREATMENTS_JSON = Path(__file__).parent.parent / "data" / "treatments.json"
CLASSES_JSON    = Path(__file__).parent.parent / "models" / "classes.json"


def _load_treatments() -> dict:
    if not TREATMENTS_JSON.exists():
        pytest.skip(f"treatments.json not found at {TREATMENTS_JSON}")
    with open(TREATMENTS_JSON, encoding="utf-8") as f:
        return json.load(f)


def _load_classes() -> list[str]:
    if not CLASSES_JSON.exists():
        pytest.skip(
            f"classes.json not found at {CLASSES_JSON}. "
            "Run train.py first or use a dummy classes.json."
        )
    with open(CLASSES_JSON) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_treatments_file_exists():
    assert TREATMENTS_JSON.exists(), (
        f"treatments.json not found at {TREATMENTS_JSON}"
    )


def test_treatments_is_valid_json():
    with open(TREATMENTS_JSON, encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)
    assert len(data) > 0


def test_each_treatment_has_required_fields():
    treatments = _load_treatments()
    required_fields = {
        "crop", "disease", "type", "symptoms",
        "organic_treatment", "chemical_treatment", "prevention", "see_expert_if"
    }
    for cls_key, entry in treatments.items():
        if cls_key.startswith("_"):
            continue  # skip meta keys like _NOTE
        missing = required_fields - set(entry.keys())
        assert not missing, (
            f"Treatment entry '{cls_key}' is missing fields: {missing}"
        )


def test_each_class_in_classes_json_has_treatment():
    """If classes.json exists, every class must have a treatment entry."""
    classes = _load_classes()
    treatments = _load_treatments()
    # Normalise keys
    treatment_keys_lower = {k.lower() for k in treatments if not k.startswith("_")}
    missing = [
        c for c in classes
        if c.lower() not in treatment_keys_lower
    ]
    assert not missing, (
        f"The following classes have no treatment entry:\n"
        + "\n".join(missing)
    )


def test_type_values_are_valid():
    treatments = _load_treatments()
    valid_types = {"fungal", "bacterial", "viral", "pest", "healthy"}
    for cls_key, entry in treatments.items():
        if cls_key.startswith("_"):
            continue
        t = entry.get("type", "")
        assert t in valid_types, (
            f"Treatment '{cls_key}' has invalid type '{t}'. "
            f"Must be one of {valid_types}."
        )
