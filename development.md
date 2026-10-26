# Development log

## 2026-10-26 - Step 1: custom CNN architecture

- Added `custom_cnn.py`, a from-scratch four-block CNN with channels 32, 64, 128, and 256.
- The model uses Conv2d, BatchNorm, ReLU, MaxPool, global average pooling, dropout 0.4, and a linear classification head.
- Added parameter-count and serialized-model-size helpers.

## 2026-10-26 - Step 2: custom CNN training workflow

- Added `scripts/train_custom_cnn.py` with fixed seeds, train-only augmentation, AdamW, cosine scheduling, label smoothing, weighted sampling, and validation-loss early stopping.
- Best checkpoints are selected by validation accuracy and saved as `models/custom_cnn.pt`; curves are saved to `models/custom_cnn_training_curves.png`.

## 2026-10-26 - Step 3: custom CNN held-out evaluation implementation

- Added `scripts/evaluate_custom_cnn.py`, which reports held-out accuracy, macro F1, per-class metrics, confusion matrix, confidence-threshold table, and the 90% PASS/FAIL result.
- The script is ready to run after custom-CNN training; no new test result was generated during this code-only update.

## 2026-10-26 - Steps 6-7: backend integration and documentation

- Added `MODEL_BACKEND` support (`pretrained` by default; `custom` after a custom checkpoint is trained) without changing the leaf gate, treatment data, or API response shape.
- Added an offline custom-backend test and documentation for training, evaluating, and selecting the custom CNN.

## 2026-10-26 - Evaluation guardrail

- Custom CNN checkpoints now record the classes used for training.
- The full-project evaluator rejects a crop-subset checkpoint before producing a misleading all-class test score.

## 2026-10-26 - Tomato-only evaluation mode

- Added explicit `--crops tomato` support to the custom-CNN evaluator.
- Tomato-only reports have distinct filenames and scope labels to prevent mixing them with 15-class metrics.

## 2026-10-26 - Dedicated Custom CNN API endpoint

- Added authenticated `POST /api/v1/predict/custom`, which lazily loads `models/custom_cnn.pt` without changing the configured default backend.
- The endpoint runs the existing CLIP leaf gate first and returns the same `not_a_leaf` response without disease or treatment when rejected.
- Added offline API tests for Custom CNN gate rejection and the normal response schema.

## 2026-10-26 - Removed comparison and field-photo utilities

- Removed the pretrained-versus-CustomCNN comparison script and the optional field-photo script.
