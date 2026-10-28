# Tomato Disease Detection

A PyTorch and FastAPI application for classifying **tomato** leaf diseases (10 classes), returning treatment guidance, and rejecting images that do not look like plant leaves before they reach the disease classifier.

## Highlights

- Fine-tuned EfficientNet-B0 disease classifier trained on **tomato classes only** (10 classes); MobileNetV3 is also supported.
- Returns crop, disease, confidence, alternative predictions, and treatment guidance.
- Uses a frozen, zero-shot CLIP ViT-B/32 leaf gate. CLIP is never trained or fine-tuned.
- Provides a protected FastAPI endpoint and offline unit tests.

## Current evaluation

The included disease model was evaluated on the held-out PlantVillage test split.

| Metric | Result |
| --- | ---: |
| Disease-model test accuracy | 96.23% |
| Disease-model macro F1 | 95.54% |
| Leaf-gate threshold | 0.9959240556 |
| Held-out non-leaf images rejected | 75/75 (100.00%) |
| Held-out real leaves wrongly rejected | 168/3,102 (5.42%) |

The gate threshold was selected only on validation images (3,095 leaf and 75 non-leaf images), then evaluated on the held-out test splits.

## Requirements

- Python 3.14, or a compatible PyTorch-supported Python version
- NVIDIA GPU optional; CUDA speeds up disease inference
- Internet access only for the first CLIP download when it is not cached

## Installation

```powershell
git clone https://github.com/Aashikhandelwal05/PLANT-DISEASE-DETECTION-.git
cd PLANT-DISEASE-DETECTION-

python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Before prediction or API use, provide these local artifacts:

```text
models/best.pt
models/classes.json
models/leaf_gate.json
data/treatments.json
```

`best.pt` and `classes.json` are written by training. `leaf_gate.json` is written by the leaf-gate tuning step below.

## Data preparation and training

Place the original, non-augmented PlantVillage images in class folders under `data/raw/PlantVillage/`, then create the stratified split:

```powershell
python scripts/prepare_data.py
```

Train the default EfficientNet-B0 model (tomato-only by default):

```powershell
python train.py
```

Useful variants:

```powershell
python train.py --crops tomato         # explicit tomato-only (same as default)
python train.py --model mobilenet      # MobileNetV3-Large backbone
python train.py --crops all            # all available crops (override default)
```

Training writes `models/best.pt`, `models/classes.json`, and `models/training_curves.png`.

### Train the custom CNN from scratch

The custom CNN has no pretrained weights. It uses the same existing `data/splits.csv` class order as the pretrained model, train-only augmentation, AdamW, cosine scheduling, label smoothing, weighted sampling, and validation-only checkpoint selection.

```powershell
python scripts/train_custom_cnn.py
```

For a quicker tomato-only experiment (same as the default):

```powershell
python scripts/train_custom_cnn.py --crops tomato
```

This produces a tomato-only checkpoint. The project is now tomato-only, so this is the standard invocation.

Evaluate a tomato-only checkpoint only on tomato test images:

```powershell
python scripts/evaluate_custom_cnn.py --crops tomato
```

This writes reports with a `custom_cnn_tomato_` prefix. They describe only the tomato classes.

This writes `models/custom_cnn.pt`, `models/custom_cnn_training_curves.png`, and `models/custom_cnn_training.json`.

### Google Colab fallback

If local training is too slow, open a Colab notebook with a GPU runtime, upload or clone this project and its PlantVillage data, then run:

```python
!pip install -r requirements.txt
!python scripts/train_custom_cnn.py
!python scripts/evaluate_custom_cnn.py
```

Copy the resulting `models/custom_cnn.pt` and `models/custom_cnn_training.json` back to your local `models/` directory.

## Leaf gate

The gate runs before disease inference. `leaf_score` is the combined CLIP softmax probability of these prompts:

- "a close-up photo of a plant leaf"
- "a photo of a tomato leaf"
- "a photo of a diseased plant leaf"
- "a photo of a green leaf on a plant"

It contrasts them with person, animal, building, vehicle, food, flower, document, landscape, and everyday-object prompts.

Prepare the non-leaf corpus, remove corrupt and duplicate files, split it 70/15/15, then tune only on validation data:

```powershell
python scripts/prepare_negatives.py --use-sklearn-digits --limit 500
python scripts/tune_leaf_gate.py
```

This creates `models/leaf_gate.json`. Only after tuning, evaluate held-out data:

```powershell
python scripts/evaluate_leaf_gate.py
python evaluate.py
```

After training the custom CNN, use the held-out split once for it:

```powershell
python scripts/evaluate_custom_cnn.py
```

The custom-CNN evaluator writes `reports/custom_cnn_test_report.txt`, `reports/custom_cnn_confusion_matrix.png`, `reports/custom_cnn_confidence_table.csv`, and `reports/custom_cnn_metrics.json`.

## Command-line prediction

```powershell
python predict.py path\to\leaf.jpg
```

Switch backend for one command, or set `MODEL_BACKEND=custom` in `.env`:

```powershell
python predict.py path\to\leaf.jpg --backend custom
python predict.py path\to\leaf.jpg --backend pretrained
```

A successful result includes the predicted class, crop, disease, confidence, alternatives, treatment advice, and leaf score. Rejected images return:

```text
This does not look like a plant leaf. Please upload a clear photo of a single leaf in good light.
```

Rejected images receive no disease prediction or treatment recommendation.

## API

Create a local `.env` file:

```text
API_KEY=replace-with-a-long-random-secret
```

Start the server:

```powershell
uvicorn api:app --reload
```

Set `MODEL_BACKEND=pretrained` (default) or `MODEL_BACKEND=custom` in `.env` before starting the API. Both backends keep the same leaf gate, treatment mapping, confidence behaviour, and response JSON shape.

Open interactive API documentation at `http://127.0.0.1:8000/docs`.

In Swagger, click **Authorize** once and enter your `X-API-Key`. Swagger then includes it automatically in every protected prediction request.

### Health check

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok","model_ready":true,"num_classes":10}
```

### Prediction request

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/predict" `
  -H "X-API-Key: replace-with-a-long-random-secret" `
  -F "file=@path\to\leaf.jpg"
```

Successful responses have `"status": "ok"`. A non-leaf response has no disease or treatment fields:

```json
{
  "status": "not_a_leaf",
  "message": "This does not look like a plant leaf. Please upload a clear photo of a single leaf in good light.",
  "leaf_score": 0.12
}
```

### Custom CNN prediction endpoint

`POST /api/v1/predict/custom` always uses `models/custom_cnn.pt`, independently of `MODEL_BACKEND`. It still runs the same CLIP leaf gate first, so a non-leaf returns the same `not_a_leaf` response and never reaches the Custom CNN.

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/predict/custom" `
  -H "X-API-Key: replace-with-a-long-random-secret" `
  -F "file=@path\to\leaf.jpg"
```

Train the Custom CNN first. If its checkpoint is unavailable, this endpoint returns HTTP 503 with an instruction to run the training script.

## Tests

Tests mock the leaf gate, so they do not download CLIP or require model weights.

```powershell
pytest -q -o addopts= --basetemp "$env:TEMP\plant-disease-pytest"
```

## Project layout

```text
app/                         Zero-shot CLIP gate
data/                        Dataset manifests and local source data
models/                      Local model artifacts (untracked)
reports/                     Local evaluation artifacts (untracked)
scripts/                     Data preparation and evaluation commands
tests/                       Offline tests
api.py                       FastAPI service
predict.py                   Command-line inference
train.py                     Disease-model training
evaluate.py                  Disease-model held-out evaluation
```

## Limitations and safety

- PlantVillage contains controlled images; real field photos may perform worse.
- The leaf gate checks whether an image appears to be a leaf, not which crop it belongs to.
- Unusual non-leaf images may pass the gate, and genuine leaf images may be rejected.
- The classifier only supports diseases represented by its trained PlantVillage classes.
- Treatment information is general guidance, not a substitute for local agricultural advice or product-label instructions.

## License

Add a license file before distributing or accepting external contributions.
