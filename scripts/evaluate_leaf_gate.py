"""Run the frozen CLIP leaf gate on the held-out leaf and non-leaf test splits."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.leaf_gate import CONFIG_PATH, LeafGate


def _test_paths(manifest: Path) -> list[Path]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        return [Path(row["file_path"]) for row in csv.DictReader(handle) if row["split"] == "test"]


def _scores(gate: LeafGate, paths: list[Path], batch_size: int = 64) -> list[float]:
    scores: list[float] = []
    for start in range(0, len(paths), batch_size):
        images = []
        for path in paths[start : start + batch_size]:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        scores.extend(gate.score_many(images))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaf-manifest", type=Path, default=Path("data/splits.csv"))
    parser.add_argument("--negative-manifest", type=Path, default=Path("data/negatives/splits.csv"))
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--out", type=Path, default=Path("reports/leaf_gate_test.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    threshold = float(config["threshold"])
    gate = LeafGate(threshold=threshold).load()
    leaf_scores = _scores(gate, _test_paths(args.leaf_manifest))
    negative_scores = _scores(gate, _test_paths(args.negative_manifest))
    result = {
        "threshold": threshold,
        "test_leaf_count": len(leaf_scores),
        "test_negative_count": len(negative_scores),
        "test_real_leaves_wrongly_rejected": sum(score < threshold for score in leaf_scores) / len(leaf_scores),
        "test_non_leaf_images_rejected": sum(score < threshold for score in negative_scores) / len(negative_scores),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
