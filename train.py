"""
train.py
========
Train EfficientNet-B0 (or MobileNetV3-Large) on the PlantVillage dataset.

Two-stage fine-tuning:
  Stage 1 — Freeze backbone, train classification head.
  Stage 2 — Unfreeze last blocks, fine-tune with lower LR.

Usage
-----
    python train.py                        # EfficientNet-B0, all crops
    python train.py --model mobilenet      # MobileNetV3-Large
    python train.py --crops tomato         # tomato classes only
    python train.py --epochs1 5 --epochs2 15
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import SEED, fix_seeds, get_device, get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Config / defaults
# ──────────────────────────────────────────────────────────────────────────────
SPLITS_CSV       = Path("data/splits.csv")
MODELS_DIR       = Path("models")
BEST_MODEL_PATH  = MODELS_DIR / "best.pt"
CLASSES_JSON     = MODELS_DIR / "classes.json"
CURVES_PNG       = MODELS_DIR / "training_curves.png"

IMG_SIZE         = 224
BATCH_SIZE       = 8
LABEL_SMOOTHING  = 0.1

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class PlantDataset(Dataset):
    def __init__(self, df: pd.DataFrame, class_to_idx: dict[str, int],
                 transform: transforms.Compose) -> None:
        self.paths   = df["file_path"].tolist()
        self.labels  = [class_to_idx[c] for c in df["class_name"].tolist()]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


def get_transforms(split: str) -> transforms.Compose:
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "mobilenet":
        model = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        )
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    else:  # efficientnet-b0 (default)
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def freeze_backbone(model: nn.Module, arch: str) -> None:
    """Freeze all parameters except the final classifier."""
    for param in model.parameters():
        param.requires_grad = False
    if arch == "mobilenet":
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        for param in model.classifier.parameters():
            param.requires_grad = True


def unfreeze_last_blocks(model: nn.Module, arch: str) -> None:
    """Unfreeze the last few feature blocks for fine-tuning."""
    if arch == "mobilenet":
        # Unfreeze last 3 InvertedResidual blocks
        for param in model.features[-3:].parameters():
            param.requires_grad = True
    else:
        # EfficientNet: unfreeze last 2 blocks + head
        for param in model.features[-2:].parameters():
            param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    train: bool,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float]:
    model.train() if train else model.eval()
    total_loss = correct = total = 0
    use_amp = device.type == "cuda" and scaler is not None

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            if train:
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def plot_curves(history: dict, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"],   label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"],   label="Val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    logger.info("Training curves saved to %s", out_path)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train plant disease classifier.")
    parser.add_argument("--model",   default="efficientnet",
                        choices=["efficientnet", "mobilenet"],
                        help="Backbone architecture.")
    parser.add_argument("--crops",   nargs="+", default=None,
                        help="Filter to specific crops, e.g. --crops tomato potato.")
    parser.add_argument("--epochs1", type=int, default=5,
                        help="Stage-1 epochs (head only).")
    parser.add_argument("--epochs2", type=int, default=20,
                        help="Stage-2 epochs (fine-tune).")
    parser.add_argument("--lr1",     type=float, default=1e-3,
                        help="Learning rate for Stage 1.")
    parser.add_argument("--lr2",     type=float, default=1e-4,
                        help="Learning rate for Stage 2.")
    parser.add_argument("--batch",   type=int, default=BATCH_SIZE)
    parser.add_argument("--patience",type=int, default=5,
                        help="Early-stopping patience (epochs without val-loss improvement).")
    args = parser.parse_args()

    fix_seeds(SEED)
    device = get_device()

    # ── Load splits ──────────────────────────────────────────────────────────
    if not SPLITS_CSV.exists():
        logger.error("splits.csv not found. Run scripts/prepare_data.py first.")
        sys.exit(1)

    df = pd.read_csv(SPLITS_CSV)

    # Optional crop filter
    if args.crops:
        crop_lower = [c.lower() for c in args.crops]
        mask = df["class_name"].str.lower().apply(
            lambda n: any(c in n for c in crop_lower)
        )
        df = df[mask].reset_index(drop=True)
        logger.info("Crop filter applied. Classes remaining: %d",
                    df["class_name"].nunique())

    classes = sorted(df["class_name"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    logger.info("Number of classes: %d", num_classes)

    # ── Save classes.json ────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CLASSES_JSON, "w") as f:
        json.dump(classes, f, indent=2)
    logger.info("classes.json saved.")

    # ── Datasets & loaders ───────────────────────────────────────────────────
    train_df = df[df["split"] == "train"]
    val_df   = df[df["split"] == "val"]

    # num_workers=0 required on Windows: CUDA + multiprocessing causes
    # 'CUDA error: unknown error' on spawned subprocesses.
    import platform
    num_workers = 0 if platform.system() == "Windows" else min(4, (torch.get_num_threads() or 1))
    logger.info("DataLoader workers: %d", num_workers)
    train_loader = DataLoader(
        PlantDataset(train_df, class_to_idx, get_transforms("train")),
        batch_size=args.batch, shuffle=True, num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        PlantDataset(val_df, class_to_idx, get_transforms("val")),
        batch_size=args.batch, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    # ── Build model ──────────────────────────────────────────────────────────
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model = build_model(args.model, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # Mixed precision scaler (FP16) — halves VRAM, speeds up GTX 1650
    scaler = torch.amp.GradScaler(device=device.type) if device.type == "cuda" else None
    logger.info("Mixed precision (AMP): %s", "enabled" if scaler else "disabled")

    history: dict[str, list] = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []
    }
    best_val_acc = 0.0

    def _run_stage(epochs: int, lr: float, stage_name: str) -> None:
        nonlocal best_val_acc
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=1e-4,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        patience_counter = 0
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = run_epoch(
                model, train_loader, criterion, optimizer, device, train=True, scaler=scaler
            )
            va_loss, va_acc = run_epoch(
                model, val_loader, criterion, optimizer, device, train=False, scaler=None
            )
            scheduler.step()
            elapsed = time.time() - t0

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(va_acc)

            logger.info(
                "[%s] Epoch %d/%d | train_loss=%.4f acc=%.3f | "
                "val_loss=%.4f acc=%.3f | %.1fs",
                stage_name, epoch, epochs, tr_loss, tr_acc, va_loss, va_acc, elapsed
            )

            # CPU time estimate on first epoch
            if device.type == "cpu" and epoch == 1:
                total_est = elapsed * epochs
                logger.info(
                    "CPU estimate: ~%.0f min per stage "
                    "(%.0fs/epoch × %d epochs).",
                    total_est / 60, elapsed, epochs,
                )

            # Save best model
            if va_acc > best_val_acc:
                best_val_acc = va_acc
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                logger.info("  → New best model saved (val_acc=%.4f)", va_acc)

            # Early stopping on val loss
            if va_loss < best_val_loss:
                best_val_loss = va_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info("Early stopping triggered after %d epochs.", epoch)
                    break

    # Stage 1: head only
    logger.info("=== Stage 1: Training classification head ===")
    freeze_backbone(model, args.model)
    _run_stage(args.epochs1, args.lr1, "Stage1")

    # Stage 2: fine-tune last blocks
    logger.info("=== Stage 2: Fine-tuning last blocks ===")
    unfreeze_last_blocks(model, args.model)
    _run_stage(args.epochs2, args.lr2, "Stage2")

    plot_curves(history, CURVES_PNG)
    logger.info("Training complete. Best val accuracy: %.4f", best_val_acc)
    logger.info("Best model saved at: %s", BEST_MODEL_PATH)


if __name__ == "__main__":
    main()
