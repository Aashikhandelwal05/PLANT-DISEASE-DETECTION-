"""Optionally compare both models on real field photos supplied by the user."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.leaf_gate import get_leaf_gate
from predict import IMAGE_EXTENSIONS, load_model, predict_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/field_test"))
    args = parser.parse_args()
    images = [path for path in args.root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS] if args.root.exists() else []
    if not images:
        print(f"SKIPPED: no field photos found under {args.root}.")
        return
    gate = get_leaf_gate()
    for backend in ("pretrained", "custom"):
        model, classes, device = load_model(backend=backend)
        correct = 0
        evaluated = 0
        for path in images:
            expected = path.parent.name
            result = predict_image(path, model, classes, device, leaf_gate=gate)
            if result.get("status") == "ok":
                evaluated += 1
                correct += result["class_name"] == expected
        accuracy = correct / evaluated if evaluated else 0.0
        print(f"{backend}: {correct}/{evaluated} classified field photos correct ({accuracy:.2%}); {len(images) - evaluated} rejected by leaf gate.")


if __name__ == "__main__":
    main()
