"""
scripts/prepare_data.py
=======================
Clean, deduplicate, and split the PlantVillage dataset.

Steps performed
---------------
1. Scan all images under --src.
2. Remove corrupt images (Pillow cannot open them).
3. Remove exact duplicates by MD5 hash (keep lexicographically first path).
4. Stratified split into TRAIN / VAL / TEST.
5. Write data/splits.csv  (columns: file_path, class_name, split).
6. Print per-class counts and verify no file appears in two splits.

Usage
-----
    python scripts/prepare_data.py
    python scripts/prepare_data.py --src data/raw --train 0.70 --val 0.15 --test 0.15
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

# Add project root to path so we can import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import SEED, get_logger

logger = get_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_images(src: Path) -> dict[str, list[Path]]:
    """Return {class_name: [image_paths]} from a directory of class sub-folders."""
    class_images: dict[str, list[Path]] = defaultdict(list)
    for class_dir in sorted(src.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                class_images[class_dir.name].append(img_path)
    return dict(class_images)


def remove_corrupt(class_images: dict[str, list[Path]]) -> dict[str, list[Path]]:
    """Remove images that Pillow cannot open. Returns cleaned dict."""
    cleaned: dict[str, list[Path]] = {}
    total_removed = 0
    for cls, paths in class_images.items():
        valid = []
        for p in paths:
            try:
                with Image.open(p) as im:
                    im.verify()
                valid.append(p)
            except (UnidentifiedImageError, Exception):
                logger.warning("Corrupt image removed: %s", p)
                total_removed += 1
        cleaned[cls] = valid
    logger.info("Corrupt images removed: %d", total_removed)
    return cleaned


def remove_duplicates(class_images: dict[str, list[Path]]) -> dict[str, list[Path]]:
    """Remove exact-duplicate images (same MD5) across all classes."""
    seen_hashes: dict[str, Path] = {}
    cleaned: dict[str, list[Path]] = defaultdict(list)
    total_removed = 0
    for cls, paths in class_images.items():
        for p in paths:
            h = _md5(p)
            if h in seen_hashes:
                logger.warning(
                    "Duplicate removed: %s  (same as %s)", p, seen_hashes[h]
                )
                total_removed += 1
            else:
                seen_hashes[h] = p
                cleaned[cls].append(p)
    logger.info("Duplicate images removed: %d", total_removed)
    return dict(cleaned)


def stratified_split(
    class_images: dict[str, list[Path]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> list[dict]:
    """Return a list of {file_path, class_name, split} dicts."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    rows: list[dict] = []
    for cls, paths in sorted(class_images.items()):
        str_paths = [str(p) for p in paths]
        labels = [cls] * len(str_paths)

        train_paths, temp_paths, _, temp_labels = train_test_split(
            str_paths, labels,
            test_size=(1.0 - train_ratio),
            stratify=labels,
            random_state=SEED,
        )
        relative_val = val_ratio / (val_ratio + test_ratio)
        val_paths, test_paths = train_test_split(
            temp_paths,
            test_size=(1.0 - relative_val),
            stratify=temp_labels,
            random_state=SEED,
        )

        for p in train_paths:
            rows.append({"file_path": p, "class_name": cls, "split": "train"})
        for p in val_paths:
            rows.append({"file_path": p, "class_name": cls, "split": "val"})
        for p in test_paths:
            rows.append({"file_path": p, "class_name": cls, "split": "test"})

    return rows


def verify_no_overlap(rows: list[dict]) -> None:
    """Assert no file appears in more than one split."""
    from collections import Counter
    counts = Counter(r["file_path"] for r in rows)
    duplicates = [p for p, c in counts.items() if c > 1]
    if duplicates:
        raise RuntimeError(
            f"Data leak detected! These files appear in multiple splits:\n"
            + "\n".join(duplicates)
        )
    logger.info("Split integrity check PASSED — no file is in multiple splits.")


def print_stats(rows: list[dict]) -> None:
    """Print per-class counts for each split."""
    from collections import defaultdict
    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        stats[r["class_name"]][r["split"]] += 1

    header = f"{'Class':<45} {'Train':>6} {'Val':>5} {'Test':>5} {'Total':>6}"
    print("\n" + header)
    print("-" * len(header))
    total_train = total_val = total_test = 0
    for cls in sorted(stats):
        t = stats[cls]["train"]
        v = stats[cls]["val"]
        te = stats[cls]["test"]
        print(f"{cls:<45} {t:>6} {v:>5} {te:>5} {t+v+te:>6}")
        total_train += t
        total_val += v
        total_test += te
    print("-" * len(header))
    grand = total_train + total_val + total_test
    print(f"{'TOTAL':<45} {total_train:>6} {total_val:>5} {total_test:>5} {grand:>6}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PlantVillage dataset splits.")
    parser.add_argument("--src", type=Path, default=Path("data/raw"),
                        help="Root of raw images (class sub-folders).")
    parser.add_argument("--out", type=Path, default=Path("data/splits.csv"),
                        help="Output CSV file path.")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val",   type=float, default=0.15)
    parser.add_argument("--test",  type=float, default=0.15)
    args = parser.parse_args()

    if not args.src.exists():
        logger.error(
            "Source directory not found: %s\n"
            "Run scripts/download_data.py first.", args.src
        )
        sys.exit(1)

    logger.info("Scanning images in %s …", args.src)
    class_images = scan_images(args.src)
    if not class_images:
        logger.error("No class sub-folders found in %s", args.src)
        sys.exit(1)
    logger.info("Found %d classes.", len(class_images))

    logger.info("Checking for corrupt images …")
    class_images = remove_corrupt(class_images)

    logger.info("Checking for duplicate images …")
    class_images = remove_duplicates(class_images)

    logger.info("Splitting data (train=%.0f%%, val=%.0f%%, test=%.0f%%) …",
                args.train * 100, args.val * 100, args.test * 100)
    rows = stratified_split(class_images, args.train, args.val, args.test)

    verify_no_overlap(rows)
    print_stats(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_path", "class_name", "split"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Splits saved to %s  (%d total images).", args.out, len(rows))


if __name__ == "__main__":
    main()
