"""
evaluate.py
===========
Run the saved model on the held-out TEST split exactly once and report results.

Outputs (saved to reports/)
---------------------------
  test_report.txt      — accuracy, macro F1, per-class precision/recall/F1
  confusion_matrix.png — confusion matrix heat-map

Usage
-----
    python evaluate.py
    python evaluate.py --model models/best.pt --splits data/splits.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from utils import fix_seeds, get_device, get_logger, SEED

logger = get_logger(__name__)

REPORTS_DIR     = Path("reports")
MODELS_DIR      = Path("models")
BEST_MODEL_PATH = MODELS_DIR / "best.pt"
CLASSES_JSON    = MODELS_DIR / "classes.json"
SPLITS_CSV      = Path("data/splits.csv")

IMG_SIZE      = 224
BATCH_SIZE    = 16
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset (reuse test-time transforms from train.py)
# ──────────────────────────────────────────────────────────────────────────────

class TestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, class_to_idx: dict[str, int]) -> None:
        self.paths  = df["file_path"].tolist()
        self.labels = [class_to_idx[c] for c in df["class_name"].tolist()]
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

def predict_all(model: nn.Module, loader: DataLoader,
                device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Return (true_labels, pred_labels)."""
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1)
            _, preds = probs.max(dim=1)
            all_true.extend(labels.numpy())
            all_pred.extend(preds.cpu().numpy())
    return (
        np.array(all_true),
        np.array(all_pred),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────────────────────────────────────

def save_text_report(
    true: np.ndarray,
    pred: np.ndarray,
    classes: list[str],
    accuracy: float,
    out_path: Path,
) -> None:
    macro_f1 = f1_score(true, pred, average="macro", zero_division=0)
    report = classification_report(
        true, pred, target_names=classes, zero_division=0
    )
    lines = [
        "=" * 60,
        "PLANT DISEASE MODEL -- TEST REPORT",
        "=" * 60,
        f"Test accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)",
        f"Macro F1      : {macro_f1:.4f}",
        "",
        "Per-class metrics:",
        report,
    ]
    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")
    # Safe print: Windows terminal may not support UTF-8
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))
    logger.info("Text report saved to %s", out_path)


def save_confusion_matrix(
    true: np.ndarray,
    pred: np.ndarray,
    classes: list[str],
    out_path: Path,
) -> None:
    cm = confusion_matrix(true, pred)
    # Normalise to percentage
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    n = len(classes)
    fig_size = max(10, n * 0.4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short = [c.split("___")[-1][:20] if "___" in c else c[:20] for c in classes]
    ax.set_xticklabels(short, rotation=90, fontsize=6)
    ax.set_yticklabels(short, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (% per true class)")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", out_path)



# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate plant disease classifier.")
    parser.add_argument("--model",  type=Path, default=BEST_MODEL_PATH)
    parser.add_argument("--splits", type=Path, default=SPLITS_CSV)
    parser.add_argument("--batch",  type=int,  default=BATCH_SIZE)
    args = parser.parse_args()

    fix_seeds(SEED)
    device = get_device()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load classes ─────────────────────────────────────────────────────────
    if not CLASSES_JSON.exists():
        logger.error("classes.json not found at %s. Run train.py first.", CLASSES_JSON)
        sys.exit(1)
    with open(CLASSES_JSON) as f:
        classes: list[str] = json.load(f)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    # ── Load model ────────────────────────────────────────────────────────────
    if not args.model.exists():
        logger.error("Model file not found: %s", args.model)
        sys.exit(1)

    # Auto-detect architecture by number of classes
    try:
        from torchvision import models as tvm
        model = tvm.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)
        state = torch.load(args.model, map_location="cpu")
        model.load_state_dict(state)
    except RuntimeError:
        from torchvision import models as tvm
        model = tvm.mobilenet_v3_large(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = torch.nn.Linear(in_features, num_classes)
        state = torch.load(args.model, map_location="cpu")
        model.load_state_dict(state)

    model = model.to(device)
    logger.info("Model loaded from %s", args.model)

    # ── Test data ─────────────────────────────────────────────────────────────
    if not args.splits.exists():
        logger.error("splits.csv not found: %s", args.splits)
        sys.exit(1)
    df = pd.read_csv(args.splits)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    logger.info("Test set size: %d images", len(test_df))

    # num_workers=0 required on Windows: CUDA + multiprocessing
    # causes MemoryError / 'CUDA error: unknown error' in worker processes.
    import platform
    num_workers = 0 if platform.system() == "Windows" else min(4, torch.get_num_threads() or 1)
    loader = DataLoader(
        TestDataset(test_df, class_to_idx),
        batch_size=args.batch, shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    # ── Run inference ────────────────────────────────────────────────────────
    true_labels, pred_labels = predict_all(model, loader, device)
    accuracy = (true_labels == pred_labels).mean()

    # ── Save reports ──────────────────────────────────────────────────────────
    save_text_report(
        true_labels, pred_labels, classes, accuracy,
        REPORTS_DIR / "test_report.txt",
    )
    save_confusion_matrix(
        true_labels, pred_labels, classes,
        REPORTS_DIR / "confusion_matrix.png",
    )
    logger.info("Evaluation complete. Test accuracy: %.4f (%.2f%%)", accuracy, accuracy * 100)



if __name__ == "__main__":
    main()
