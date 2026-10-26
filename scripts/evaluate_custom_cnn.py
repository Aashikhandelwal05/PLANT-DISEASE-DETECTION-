"""Evaluate the trained CustomCNN once on the existing held-out test split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))
from custom_cnn import CustomCNN, model_size_mb, parameter_count
from utils import get_device

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CustomCNN on the held-out test split.")
    parser.add_argument("--model", type=Path, default=Path("models/custom_cnn.pt"))
    parser.add_argument("--classes", type=Path, default=Path("models/classes.json"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits.csv"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Custom CNN checkpoint not found: {args.model}")
    classes = json.loads(args.classes.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.model, map_location="cpu")
    if checkpoint.get("classes") != classes:
        raise ValueError("Custom CNN checkpoint classes differ from models/classes.json.")
    if "trained_classes" not in checkpoint:
        raise ValueError(
            "This custom checkpoint lacks training-class metadata. Retrain it with "
            "the current scripts/train_custom_cnn.py before evaluating."
        )
    trained_classes = set(checkpoint["trained_classes"])
    if trained_classes != set(classes):
        missing = sorted(set(classes) - trained_classes)
        raise ValueError(
            "This checkpoint was trained on a subset of classes and cannot be "
            "evaluated against the full 15-class test split. Missing classes: "
            f"{', '.join(missing)}. Train without --crops for the project-wide "
            "comparison, or create a matching subset evaluator."
        )
    model = CustomCNN(len(classes), int(checkpoint.get("input_size", 128)))
    model.load_state_dict(checkpoint["state_dict"])
    device = get_device()
    model = model.to(device).eval()
    size = int(checkpoint.get("input_size", 128))
    transform = transforms.Compose([
        transforms.Resize(int(size * 1.15)), transforms.CenterCrop(size),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    test_rows = [row for row in csv.DictReader(args.splits.open(newline="", encoding="utf-8")) if row["split"] == "test"]
    class_to_idx = {name: index for index, name in enumerate(classes)}
    true, predicted, confidence = [], [], []
    started = time.perf_counter()
    with torch.no_grad():
        for row in test_rows:
            with Image.open(row["file_path"]) as image:
                tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
            probabilities = torch.softmax(model(tensor), dim=1)[0].cpu()
            score, index = probabilities.max(0)
            true.append(class_to_idx[row["class_name"]])
            predicted.append(int(index))
            confidence.append(float(score))
    elapsed = time.perf_counter() - started
    accuracy = float(np.mean(np.array(true) == np.array(predicted)))
    macro_f1 = float(f1_score(true, predicted, average="macro", zero_division=0))
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    report = classification_report(true, predicted, target_names=classes, zero_division=0)
    text = f"CustomCNN test accuracy: {accuracy:.6f} ({accuracy * 100:.2f}%)\nMacro F1: {macro_f1:.6f}\n\n{report}"
    (args.reports_dir / "custom_cnn_test_report.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"{'PASS' if accuracy > 0.90 else 'FAIL'}: test accuracy above 90% = {accuracy > 0.90}")
    thresholds = (0.5, 0.7, 0.8, 0.9)
    table = []
    for threshold in thresholds:
        selected = [index for index, score in enumerate(confidence) if score >= threshold]
        selected_accuracy = float(np.mean([true[index] == predicted[index] for index in selected])) if selected else None
        table.append({"confidence_threshold": threshold, "share_of_images": len(selected) / len(true), "accuracy": selected_accuracy})
    with (args.reports_dir / "custom_cnn_confidence_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=table[0].keys())
        writer.writeheader(); writer.writerows(table)
    matrix = confusion_matrix(true, predicted)
    fig, axis = plt.subplots(figsize=(12, 10))
    axis.imshow(matrix, cmap="Blues")
    axis.set(title="CustomCNN confusion matrix", xlabel="Predicted", ylabel="True")
    fig.tight_layout(); fig.savefig(args.reports_dir / "custom_cnn_confusion_matrix.png", dpi=140); plt.close(fig)
    metrics = {
        "accuracy": accuracy, "macro_f1": macro_f1, "test_images": len(true),
        "parameter_count": parameter_count(model), "model_size_mb": model_size_mb(model),
        "inference_seconds_per_image": elapsed / len(true), "confidence_table": table,
    }
    (args.reports_dir / "custom_cnn_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
