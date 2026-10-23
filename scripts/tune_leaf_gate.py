"""Tune the frozen CLIP leaf gate on validation images only."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.leaf_gate import CONFIG_PATH, LEAF_PROMPTS, MODEL_NAME, NON_LEAF_PROMPTS, LeafGate


def _paths(manifest: Path, split: str) -> list[Path]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        return [Path(row["file_path"]) for row in csv.DictReader(handle) if row["split"] == split]


def _scores(gate: LeafGate, paths: list[Path], batch_size: int = 64) -> list[float]:
    values: list[float] = []
    for start in range(0, len(paths), batch_size):
        images = []
        for path in paths[start : start + batch_size]:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        values.extend(gate.score_many(images))
    return values


def _threshold_with_95_percent_leaf_recall(leaf_scores: list[float]) -> float:
    if not leaf_scores:
        raise ValueError("No validation leaf images were found.")
    # The fifth percentile is the largest threshold that preserves >=95% recall.
    ordered = sorted(leaf_scores)
    return ordered[max(0, int(len(ordered) * 0.05) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaf-manifest", type=Path, default=Path("data/splits.csv"))
    parser.add_argument("--negative-manifest", type=Path, default=Path("data/negatives/splits.csv"))
    parser.add_argument("--out", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()

    leaves = _paths(args.leaf_manifest, "val")
    negatives = _paths(args.negative_manifest, "val")
    gate = LeafGate().load()
    leaf_scores = _scores(gate, leaves)
    negative_scores = _scores(gate, negatives)
    threshold = _threshold_with_95_percent_leaf_recall(leaf_scores)
    payload = {
        "model_name": MODEL_NAME,
        "threshold": threshold,
        "leaf_prompts": list(LEAF_PROMPTS),
        "non_leaf_prompts": list(NON_LEAF_PROMPTS),
        "tuning_split": "validation",
        "validation_leaf_count": len(leaf_scores),
        "validation_negative_count": len(negative_scores),
        "validation_leaf_acceptance": sum(score >= threshold for score in leaf_scores) / len(leaf_scores),
        "validation_negative_rejection": sum(score < threshold for score in negative_scores) / len(negative_scores),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
