# Plant Leaf Disease Detection

A transfer-learning system that classifies plant leaf photos into disease categories (or healthy), reports confidence, and shows treatment information. Built with PyTorch and the PlantVillage dataset.

---

## What It Does

- Classifies a leaf photo into one of 38 disease/healthy classes (PlantVillage, all crops) or 10 classes (tomato only).
- Reports confidence as a percentage; flags low-confidence predictions (< 90%).
- Shows treatment information: symptoms, organic/chemical options, prevention, and when to see an expert.
- Provides a REST API (FastAPI) for integration into other tools.

---

## Model Performance

| Split | Accuracy | Macro F1 |
|-------|----------|----------|
| Validation | — | — |
| **Test** | **—** | **—** |

*(Fill in after training and evaluation.)*

---

## Setup

### Requirements
- Python 3.10 or 3.11
- Windows PowerShell (or any OS terminal)
- Internet connection for dataset download and initial model weights
- GPU optional; CPU-only is supported (training will be slow — see note below)

### Install

```powershell
# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

> **CPU-only note:** Training on CPU takes several hours. See the [Google Colab Fallback](#google-colab-fallback) section below.

---

## Dataset Download

You need the **original (non-augmented) PlantVillage colour images**.

### Option A — Kaggle (recommended)

1. Create a free account at https://www.kaggle.com
2. Go to Account → API → Create New API Token → download `kaggle.json`
3. Place `kaggle.json` at `C:\Users\<you>\.kaggle\kaggle.json` (Windows)
4. Run:

```powershell
python scripts/download_data.py
```

The script will download and extract the dataset to `data/raw/`.

### Option B — Manual Download

Download the dataset from Kaggle:
https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

Extract the `color/` folder to `data/raw/PlantVillage/`.

> **⚠ Warning:** If you only have the **augmented** version of PlantVillage, the script will warn you. Augmented copies of the same image can land in different splits, which inflates accuracy. Use the `color/` folder only.

---

## Prepare Data

```powershell
# Default: all crops, 70/15/15 split
python scripts/prepare_data.py

# Tomato only
python scripts/prepare_data.py --crops tomato

# Custom ratios
python scripts/prepare_data.py --train 0.7 --val 0.15 --test 0.15
```

Outputs:
- `data/splits.csv` — every image path, class, and split assignment
- Console: image counts per class per split

---

## Training

```powershell
# All crops (default, EfficientNet-B0)
python train.py

# Tomato only
python train.py --crops tomato

# Use MobileNetV3 (faster on slow CPU)
python train.py --model mobilenet
```

Outputs saved to `models/`:
- `best.pt` — best model weights (by validation accuracy)
- `classes.json` — ordered list of class names
- `training_curves.png` — loss and accuracy curves

---

## Evaluation (Run Once on Test Set)

```powershell
python evaluate.py
```

Outputs saved to `reports/`:
- `test_report.txt` — accuracy, macro F1, per-class metrics
- `confusion_matrix.png`
- `confidence_table.txt`
- PASS or FAIL for the 90% accuracy target

> **Important:** Run this only once, after training is complete. The test set must never be used to tune anything.

---

## Prediction

```powershell
python predict.py path/to/leaf.jpg
```

Example output:
```
Crop:      Tomato
Disease:   Early Blight
Confidence: 94.3%

Top 3 guesses:
  1. Tomato__Early_blight       94.3%
  2. Tomato__Late_blight         3.1%
  3. Tomato__healthy             1.4%

Treatment:
  Symptoms: Dark concentric rings on lower/older leaves...
  Organic:  Remove infected leaves; apply copper-based fungicide...
  Chemical: Chlorothalonil or mancozeb (follow product label)...
  Prevention: Crop rotation, avoid overhead watering...
  See an expert if: Disease spreads rapidly or affects >30% of the plant.
```

If confidence is below 90%:
```
Not sure (confidence 62.1%). Please retake a clear photo of a single leaf in good light.
```

---

## API

Start the server:
```powershell
# Create .env with your API key
echo "API_KEY=change-me-before-use" > .env

uvicorn api:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `POST /api/v1/predict` — Upload an image, get disease prediction + treatment
- `GET /health` — Health check
- `GET /docs` — Swagger UI

Example request:
```powershell
curl -X POST "http://localhost:8000/api/v1/predict" `
  -H "X-API-Key: change-me-before-use" `
  -F "file=@leaf.jpg"
```

---

## Running Tests

```powershell
pytest tests/ -v
```

Tests run offline without the real dataset or a trained model.

---

## Google Colab Fallback

If training on CPU is too slow (estimate: 8–15 hours for all crops):

1. Upload your project to Google Drive or clone from GitHub.
2. Open a new Colab notebook with a **GPU runtime** (Runtime → Change runtime type → T4 GPU).
3. Mount Drive and run:

```python
!pip install -r requirements.txt
!python scripts/download_data.py   # needs kaggle.json in /root/.kaggle/
!python scripts/prepare_data.py
!python train.py
```

4. Download `models/best.pt` and `models/classes.json` back to your machine.

---

## Project Structure

```
plant-disease/
├── data/                  # ← not committed (large)
│   ├── raw/               # downloaded dataset
│   ├── splits.csv         # train/val/test assignments
│   └── treatments.json    # treatment info per class
├── models/                # ← not committed (large)
│   ├── best.pt
│   └── classes.json
├── reports/               # evaluation outputs
├── scripts/
│   ├── download_data.py
│   └── prepare_data.py
├── tests/
├── train.py
├── evaluate.py
├── predict.py
├── api.py
├── requirements.txt
├── README.md
└── development.md
```

---

## Limitations

- **Controlled photos:** The PlantVillage dataset consists of photos taken against plain backgrounds in controlled conditions. Accuracy on real field photos (cluttered background, variable lighting, multiple leaves) will likely be **lower** than reported numbers.
- **Leaf check:** The model does not verify that the uploaded image actually shows a leaf. Non-leaf images will still receive a prediction.
- **Crop scope:** Only covers crops present in PlantVillage. It will not recognise diseases of crops it was not trained on.
- **Treatment guidance:** Treatment information is general. Always follow product labels and local agricultural advice. Consult an agricultural expert before applying any treatment.
- **Not a substitute for expert diagnosis:** This tool is a screening aid, not a certified diagnostic system.
