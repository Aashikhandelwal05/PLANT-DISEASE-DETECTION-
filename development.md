# Development log

## 2026-10-26 — Step 1: custom CNN architecture

- Added `custom_cnn.py`, a from-scratch four-block CNN with channels 32, 64, 128, and 256.
- The model uses Conv2d, BatchNorm, ReLU, MaxPool, global average pooling, dropout 0.4, and a linear classification head.
- Added parameter-count and serialized-model-size helpers.

## 2026-10-26 — Step 2: custom CNN training workflow

- Added `scripts/train_custom_cnn.py` with fixed seeds, train-only augmentation, AdamW, cosine scheduling, label smoothing, weighted sampling, and validation-loss early stopping.
- Best checkpoints are selected by validation accuracy and saved as `models/custom_cnn.pt`; curves are saved to `models/custom_cnn_training_curves.png`.
