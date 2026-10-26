"""Train the from-scratch CustomCNN using the existing immutable split CSV."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))
from custom_cnn import CustomCNN, describe_model
from utils import SEED, fix_seeds, get_device, get_logger

logger = get_logger(__name__)
SPLITS_PATH = Path("data/splits.csv")
CLASSES_PATH = Path("models/classes.json")
MODEL_PATH = Path("models/custom_cnn.pt")
CURVES_PATH = Path("models/custom_cnn_training_curves.png")
TRAINING_METADATA_PATH = Path("models/custom_cnn_training.json")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class PlantDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, class_to_idx: dict[str, int], transform) -> None:
        self.paths = [Path(path) for path in frame.file_path.tolist()]
        self.labels = [class_to_idx[name] for name in frame.class_name.tolist()]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB")), self.labels[index]


def transforms_for(input_size: int):
    train = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    evaluate = transforms.Compose([
        transforms.Resize(int(input_size * 1.15)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train, evaluate


def filter_crops(frame: pd.DataFrame, crops: list[str] | None) -> pd.DataFrame:
    if not crops:
        return frame
    prefixes = tuple(crop.lower() for crop in crops)
    selected = frame[frame.class_name.str.lower().str.startswith(prefixes)].copy()
    if selected.empty:
        raise ValueError(f"No classes match --crops {' '.join(crops)}")
    return selected


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
    return loss_sum / len(loader.dataset), correct / len(loader.dataset)


def save_curves(history: dict[str, list[float]], path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()
    axes[1].plot(epochs, history["train_accuracy"], label="train")
    axes[1].plot(epochs, history["val_accuracy"], label="validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CustomCNN from scratch.")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--crops", nargs="+", help="Optional crop prefixes, e.g. --crops tomato")
    args = parser.parse_args()

    fix_seeds(SEED)
    training_started = time.perf_counter()
    if not SPLITS_PATH.exists() or not CLASSES_PATH.exists():
        raise FileNotFoundError("Expected data/splits.csv and models/classes.json before training.")
    classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    class_to_idx = {name: index for index, name in enumerate(classes)}
    frame = filter_crops(pd.read_csv(SPLITS_PATH), args.crops)
    train_frame = frame[frame.split == "train"].reset_index(drop=True)
    val_frame = frame[frame.split == "val"].reset_index(drop=True)
    if train_frame.empty or val_frame.empty:
        raise ValueError("Selected crops must have both train and validation images.")

    train_transform, eval_transform = transforms_for(args.input_size)
    train_set = PlantDataset(train_frame, class_to_idx, train_transform)
    val_set = PlantDataset(val_frame, class_to_idx, eval_transform)
    counts = Counter(train_set.labels)
    sample_weights = torch.DoubleTensor([1.0 / counts[label] for label in train_set.labels])
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True, generator=torch.Generator().manual_seed(SEED))
    device = get_device()
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=args.batch, sampler=sampler, num_workers=0, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=pin_memory)

    model = CustomCNN(len(classes), args.input_size).to(device)
    print(describe_model(model))
    print(f"Training {len(train_set):,} images; validating {len(val_set):,} images; device={device}.")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = {name: [] for name in ("train_loss", "val_loss", "train_accuracy", "val_accuracy")}
    best_accuracy = -1.0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        loss_sum = 0.0
        correct = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
        train_loss = loss_sum / len(train_set)
        train_accuracy = correct / len(train_set)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        for name, value in (("train_loss", train_loss), ("val_loss", val_loss), ("train_accuracy", train_accuracy), ("val_accuracy", val_accuracy)):
            history[name].append(value)
        elapsed = time.perf_counter() - started
        if device.type == "cpu" and epoch == 1:
            print(f"CPU first epoch: {elapsed:.1f}s; estimated {elapsed * args.epochs / 60:.1f} minutes for {args.epochs} epochs.")
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.4f} acc {train_accuracy:.4f} | val loss {val_loss:.4f} acc {val_accuracy:.4f} | {elapsed:.1f}s")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            stale_epochs = 0
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "input_size": args.input_size, "num_classes": len(classes), "classes": classes}, MODEL_PATH)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs; best validation accuracy {best_accuracy:.4f}.")
                break
    save_curves(history, CURVES_PATH)
    TRAINING_METADATA_PATH.write_text(json.dumps({
        "best_validation_accuracy": best_accuracy,
        "epochs_completed": len(history["train_loss"]),
        "training_seconds": time.perf_counter() - training_started,
        "input_size": args.input_size,
        "classes": classes,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved best validation checkpoint to {MODEL_PATH}; best validation accuracy {best_accuracy:.4f}.")


if __name__ == "__main__":
    main()
