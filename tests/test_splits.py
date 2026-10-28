"""
tests/test_splits.py
====================
Tests for the data split logic in scripts/prepare_data.py.
Runs entirely offline using synthetic data (no real images required).
"""

from __future__ import annotations

import sys
import csv
import json
import tempfile
from collections import Counter
from pathlib import Path

import pytest

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from prepare_data import stratified_split, verify_no_overlap


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def fake_class_images(tmp_path: Path) -> dict[str, list[Path]]:
    """
    Build a fake class_images dict with 3 classes, 30 images each.
    Files are not real images — tests only check path strings.
    """
    classes = ["Tomato_healthy", "Tomato_Early_blight", "Tomato_Late_blight"]
    data: dict[str, list[Path]] = {}
    for cls in classes:
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        data[cls] = [cls_dir / f"img_{i:03d}.jpg" for i in range(30)]
        for p in data[cls]:
            p.touch()
    return data


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_split_produces_all_splits(fake_class_images):
    rows = stratified_split(fake_class_images, 0.70, 0.15, 0.15)
    splits_found = {r["split"] for r in rows}
    assert splits_found == {"train", "val", "test"}, (
        f"Expected train/val/test splits, got: {splits_found}"
    )


def test_no_overlap(fake_class_images):
    rows = stratified_split(fake_class_images, 0.70, 0.15, 0.15)
    # verify_no_overlap raises RuntimeError on data leak
    verify_no_overlap(rows)


def test_all_classes_present(fake_class_images):
    rows = stratified_split(fake_class_images, 0.70, 0.15, 0.15)
    classes_in_rows = {r["class_name"] for r in rows}
    assert classes_in_rows == set(fake_class_images.keys())


def test_total_images_unchanged(fake_class_images):
    total_before = sum(len(v) for v in fake_class_images.values())
    rows = stratified_split(fake_class_images, 0.70, 0.15, 0.15)
    assert len(rows) == total_before


def test_train_ratio_approx(fake_class_images):
    rows = stratified_split(fake_class_images, 0.70, 0.15, 0.15)
    counter = Counter(r["split"] for r in rows)
    total = len(rows)
    train_ratio = counter["train"] / total
    # Allow ±5% tolerance per class rounding
    assert 0.65 <= train_ratio <= 0.75, f"Train ratio {train_ratio:.2f} out of range."


def test_stratified_preserves_class_balance(fake_class_images):
    """Each class should appear in each split."""
    rows = stratified_split(fake_class_images, 0.70, 0.15, 0.15)
    for cls in fake_class_images:
        cls_rows = [r for r in rows if r["class_name"] == cls]
        splits = {r["split"] for r in cls_rows}
        assert {"train", "val", "test"} == splits, (
            f"Class {cls} missing from some splits: {splits}"
        )
