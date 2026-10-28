"""A compact CNN trained from scratch for PlantVillage classification."""

from __future__ import annotations

import torch
from torch import nn


class CustomCNN(nn.Module):
    """Six Conv-BatchNorm-ReLU-MaxPool blocks followed by a linear head."""

    def __init__(self, num_classes: int, input_size: int = 128, dropout: float = 0.5):
        super().__init__()
        if input_size < 16:
            raise ValueError("input_size must be at least 16 pixels.")
        channels = (32, 64, 128, 256, 512, 512)
        layers: list[nn.Module] = []
        in_channels = 3
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(self.dropout(x))


def parameter_count(model: nn.Module) -> int:
    """Return all learnable parameters in a model."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def model_size_mb(model: nn.Module) -> float:
    """Return serialized parameter and buffer size in MiB."""
    byte_count = sum(tensor.numel() * tensor.element_size() for tensor in model.state_dict().values())
    return byte_count / (1024 * 1024)


def describe_model(model: nn.Module) -> str:
    return f"Parameters: {parameter_count(model):,} | Model size: {model_size_mb(model):.2f} MiB"
