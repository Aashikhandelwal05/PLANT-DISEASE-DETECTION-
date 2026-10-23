"""
Shared utilities: random seed fixing, device selection, logging helpers.
"""

from __future__ import annotations

import random
import os
import logging

import numpy as np
import torch

# ──────────────────────────────────────────────
# Shared constant
# ──────────────────────────────────────────────
SEED: int = 42


def fix_seeds(seed: int = SEED) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic ops where possible (may slow things down slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Return the best available device and print a message."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(device)
        print(f"[device] GPU detected: {props.name} "
              f"({props.total_memory // 1024**2} MB VRAM)")
    else:
        device = torch.device("cpu")
        print("[device] No GPU found — running on CPU. "
              "Training will be slow. See README for Google Colab fallback.")
    return device


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a simple console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
