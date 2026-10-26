"""Create a reproducible pretrained-versus-CustomCNN comparison report."""

from __future__ import annotations

import json
import argparse
import re
import sys
import time
from pathlib import Path

import torch
from torchvision import models

sys.path.insert(0, str(Path(__file__).parent.parent))
from custom_cnn import CustomCNN, model_size_mb, parameter_count


def _metrics_from_pretrained_report(path: Path) -> tuple[float, float]:
    text = path.read_text(encoding="utf-8")
    accuracy = re.search(r"Test accuracy\s*:\s*([0-9.]+)", text)
    macro_f1 = re.search(r"Macro F1\s*:\s*([0-9.]+)", text)
    if not accuracy or not macro_f1:
        raise ValueError(f"Could not read accuracy and macro F1 from {path}.")
    return float(accuracy.group(1)), float(macro_f1.group(1))


def _cpu_time(model: torch.nn.Module, input_size: int, iterations: int = 30) -> float:
    model = model.cpu().eval()
    sample = torch.zeros(1, 3, input_size, input_size)
    with torch.no_grad():
        for _ in range(5): model(sample)
        started = time.perf_counter()
        for _ in range(iterations): model(sample)
    return (time.perf_counter() - started) / iterations


def main() -> None:
    argparse.ArgumentParser(description="Create reports/comparison.md after both test evaluations.").parse_args()
    reports = Path("reports")
    classes = json.loads(Path("models/classes.json").read_text(encoding="utf-8"))
    custom_metrics = json.loads((reports / "custom_cnn_all_classes_metrics.json").read_text(encoding="utf-8"))
    if custom_metrics.get("scope") != "all_classes":
        raise ValueError("reports/custom_cnn_all_classes_metrics.json must contain a full 15-class evaluation.")
    training = json.loads(Path("models/custom_cnn_training.json").read_text(encoding="utf-8"))
    pretrained_accuracy, pretrained_f1 = _metrics_from_pretrained_report(reports / "test_report.txt")
    pretrained = models.efficientnet_b0(weights=None)
    pretrained.classifier[1] = torch.nn.Linear(pretrained.classifier[1].in_features, len(classes))
    pretrained.load_state_dict(torch.load("models/best.pt", map_location="cpu"))
    custom_checkpoint = torch.load("models/custom_cnn.pt", map_location="cpu")
    custom = CustomCNN(len(classes), int(custom_checkpoint.get("input_size", 128)))
    custom.load_state_dict(custom_checkpoint["state_dict"])
    rows = [
        ("Pretrained EfficientNet-B0", pretrained_accuracy, pretrained_f1, parameter_count(pretrained), model_size_mb(pretrained), _cpu_time(pretrained, 224), "Not recorded"),
        ("Custom CNN", custom_metrics["accuracy"], custom_metrics["macro_f1"], parameter_count(custom), model_size_mb(custom), _cpu_time(custom, int(custom_checkpoint.get("input_size", 128))), f"{training['training_seconds']:.1f} s"),
    ]
    lines = ["# Model comparison", "", "All accuracy and macro-F1 values are from the same held-out `data/splits.csv` test split.", "", "| Model | Accuracy | Macro F1 | Parameters | Size (MiB) | CPU inference / image | Training time |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, accuracy, f1, params, size, seconds, training_time in rows:
        lines.append(f"| {name} | {accuracy:.4f} | {f1:.4f} | {params:,} | {size:.2f} | {seconds * 1000:.2f} ms | {training_time} |")
    (reports / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
