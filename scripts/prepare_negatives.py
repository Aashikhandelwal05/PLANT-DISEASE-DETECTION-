"""Build a cleaned, deduplicated 70/15/15 non-leaf image manifest.

The optional CIFAR-10 source is used only as a public non-leaf corpus.  Its
training split is exported before any gate tuning so that no test image can
influence threshold selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
SEED = 42


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _export_cifar10(destination: Path, limit: int) -> None:
    """Download CIFAR-10 and export a deterministic non-leaf subset as PNG."""
    from torchvision.datasets import CIFAR10

    dataset = CIFAR10(root=str(destination / "_cifar10"), train=True, download=True)
    image_dir = destination / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for index in range(min(limit, len(dataset))):
        image, label = dataset[index]
        image.save(image_dir / f"cifar10_{index:05d}_class{label}.png")


def _export_sklearn_digits(destination: Path, limit: int) -> None:
    """Export public handwritten-digit images as non-leaf document examples.

    This is an offline fallback for environments where downloading the optional
    CIFAR-10 corpus is unavailable.  The images come from the public UCI
    optical-recognition data distributed with scikit-learn.
    """
    from sklearn.datasets import load_digits

    dataset = load_digits()
    image_dir = destination / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for index, (pixels, label) in enumerate(zip(dataset.images[:limit], dataset.target[:limit])):
        image = Image.fromarray((pixels / pixels.max() * 255).astype("uint8"), mode="L")
        image.convert("RGB").save(image_dir / f"sklearn_digits_{index:05d}_class{label}.png")


def _clean_and_deduplicate(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    clean: list[Path] = []
    for path in sorted(paths):
        try:
            with Image.open(path) as image:
                image.verify()
        except (UnidentifiedImageError, OSError):
            continue
        digest = _md5(path)
        if digest not in seen:
            seen.add(digest)
            clean.append(path)
    return clean


def build_manifest(root: Path, output: Path) -> int:
    candidates = [
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        and "_cifar10" not in path.parts
    ]
    paths = _clean_and_deduplicate(candidates)
    if len(paths) < 20:
        raise RuntimeError("At least 20 cleaned negative images are required.")

    train, temporary = train_test_split(paths, test_size=0.30, random_state=SEED)
    validation, test = train_test_split(temporary, test_size=0.50, random_state=SEED)
    rows = (
        [{"file_path": str(path), "split": "train"} for path in train]
        + [{"file_path": str(path), "split": "val"} for path in validation]
        + [{"file_path": str(path), "split": "test"} for path in test]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file_path", "split"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/negatives"))
    parser.add_argument("--out", type=Path, default=Path("data/negatives/splits.csv"))
    parser.add_argument("--download-cifar10", action="store_true")
    parser.add_argument("--use-sklearn-digits", action="store_true")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    if args.download_cifar10:
        _export_cifar10(args.root, args.limit)
    if args.use_sklearn_digits:
        _export_sklearn_digits(args.root, args.limit)
    count = build_manifest(args.root, args.out)
    print(f"Saved {count} cleaned, deduplicated negative-image assignments to {args.out}")


if __name__ == "__main__":
    main()
