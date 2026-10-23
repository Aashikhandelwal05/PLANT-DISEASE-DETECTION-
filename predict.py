"""
predict.py
==========
Command-line prediction tool for plant leaf disease detection.

Usage
-----
    python predict.py <image_path>
    python predict.py leaf.jpg
    python predict.py path/to/leaf.png --top 5

Output
------
    Crop          : Tomato
    Disease       : Early Blight
    Confidence    : 95.32%

    Treatment advice:
      ...

    Top-3 guesses:
      1. Tomato___Early_blight           95.32%
      2. Tomato___Septoria_leaf_spot      3.21%
      3. Tomato___healthy                 0.81%
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))
from utils import fix_seeds, get_device, get_logger, SEED

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
MODELS_DIR      = Path("models")
BEST_MODEL_PATH = MODELS_DIR / "best.pt"
CLASSES_JSON    = MODELS_DIR / "classes.json"
TREATMENTS_JSON = Path("data/treatments.json")

IMG_SIZE      = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ──────────────────────────────────────────────────────────────────────────────
# Model loading (cached at module level if called multiple times)
# ──────────────────────────────────────────────────────────────────────────────

_model_cache: dict = {}


def load_model(
    model_path: Path = BEST_MODEL_PATH,
    classes_path: Path = CLASSES_JSON,
    device: torch.device | None = None,
):
    """Load model and class list, returning (model, classes, device)."""
    if "model" in _model_cache:
        return _model_cache["model"], _model_cache["classes"], _model_cache["device"]

    if device is None:
        device = get_device()

    if not classes_path.exists():
        raise FileNotFoundError(
            f"classes.json not found at {classes_path}. Run train.py first."
        )
    with open(classes_path) as f:
        classes: list[str] = json.load(f)

    num_classes = len(classes)

    # Try EfficientNet-B0 first, fall back to MobileNetV3
    try:
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
    except RuntimeError:
        model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = torch.nn.Linear(in_features, num_classes)
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)

    model = model.to(device).eval()
    _model_cache["model"] = model
    _model_cache["classes"] = classes
    _model_cache["device"] = device
    return model, classes, device


# ──────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────────────────────

def predict_image(
    image_path: Path,
    model: torch.nn.Module,
    classes: list[str],
    device: torch.device,
    top_k: int = 3,
) -> dict:
    """
    Run inference on a single image.

    Returns a dict with keys:
        class_name, crop, disease, confidence, confident,
        top_k (list of {class_name, confidence}), treatment
    """
    # Validate file type
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: '{image_path.suffix}'. "
            f"Supported: {', '.join(sorted(IMAGE_EXTENSIONS))}"
        )

    # Load image
    try:
        img = Image.open(image_path).convert("RGB")
    except UnidentifiedImageError:
        raise ValueError(f"Cannot open image file: {image_path}")

    tensor = TRANSFORM(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top_probs, top_idx = probs.topk(min(top_k, len(classes)))
    top_probs = top_probs.cpu().tolist()
    top_idx   = top_idx.cpu().tolist()

    best_class = classes[top_idx[0]]
    best_conf  = top_probs[0]

    # Hardcoded crop/disease lookup — PlantVillage class names use
    # inconsistent separators (_  __  ___) so string-splitting is unreliable.
    _CLASS_LABELS: dict[str, tuple[str, str]] = {
        "Pepper__bell___Bacterial_spot":                    ("Pepper (Bell)",  "Bacterial Spot"),
        "Pepper__bell___healthy":                           ("Pepper (Bell)",  "Healthy"),
        "Potato___Early_blight":                            ("Potato",         "Early Blight"),
        "Potato___Late_blight":                             ("Potato",         "Late Blight"),
        "Potato___healthy":                                 ("Potato",         "Healthy"),
        "Tomato_Bacterial_spot":                            ("Tomato",         "Bacterial Spot"),
        "Tomato_Early_blight":                              ("Tomato",         "Early Blight"),
        "Tomato_Late_blight":                               ("Tomato",         "Late Blight"),
        "Tomato_Leaf_Mold":                                 ("Tomato",         "Leaf Mold"),
        "Tomato_Septoria_leaf_spot":                        ("Tomato",         "Septoria Leaf Spot"),
        "Tomato_Spider_mites_Two_spotted_spider_mite":      ("Tomato",         "Spider Mites (Two-spotted)"),
        "Tomato__Target_Spot":                              ("Tomato",         "Target Spot"),
        "Tomato__Tomato_YellowLeaf__Curl_Virus":            ("Tomato",         "Yellow Leaf Curl Virus"),
        "Tomato__Tomato_mosaic_virus":                      ("Tomato",         "Mosaic Virus"),
        "Tomato_healthy":                                   ("Tomato",         "Healthy"),
    }
    crop, disease = _CLASS_LABELS.get(best_class, (best_class.replace("_", " "), "Unknown"))

    # Load treatment
    treatment = _get_treatment(best_class)

    return {
        "class_name": best_class,
        "crop":       crop,
        "disease":    disease,
        "confidence": best_conf,
        "top_k": [
            {"class_name": classes[i], "confidence": p}
            for i, p in zip(top_idx, top_probs)
        ],
        "treatment": treatment,
    }


def _get_treatment(class_name: str) -> dict:
    """Return treatment dict for class_name, or a fallback message."""
    if not TREATMENTS_JSON.exists():
        return {"note": "treatments.json not found."}
    with open(TREATMENTS_JSON, encoding="utf-8") as f:
        treatments: dict = json.load(f)
    # Try exact match first, then case-insensitive
    if class_name in treatments:
        return treatments[class_name]
    for key, val in treatments.items():
        if key.lower() == class_name.lower():
            return val
    return {
        "note": (
            f"No treatment entry found for '{class_name}'. "
            "Check data/treatments.json."
        )
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI display
# ──────────────────────────────────────────────────────────────────────────────

def _format_result(result: dict) -> str:
    lines = [""]
    conf_pct = result["confidence"] * 100

    lines.append(f"{'Crop':<16}: {result['crop']}")
    lines.append(f"{'Disease':<16}: {result['disease']}")
    lines.append(f"{'Confidence':<16}: {conf_pct:.2f}%")

    t = result.get("treatment", {})
    if t and "note" not in t:
        lines.append("")
        lines.append("-- Treatment advice " + "-" * 42)
        lines.append(f"  Type        : {t.get('type', 'N/A').capitalize()}")
        lines.append(f"  Symptoms    : {t.get('symptoms', 'N/A')}")
        lines.append(f"  Organic     : {t.get('organic_treatment', 'N/A')}")
        lines.append(f"  Chemical    : {t.get('chemical_treatment', 'N/A')}")
        lines.append(f"  Prevention  : {t.get('prevention', 'N/A')}")
        lines.append(f"  See expert  : {t.get('see_expert_if', 'N/A')}")
        lines.append("-" * 60)
    elif t:
        lines.append(f"  Note: {t.get('note', '')}")

    lines.append("")
    lines.append(f"Top-{len(result['top_k'])} guesses:")
    for i, entry in enumerate(result["top_k"], 1):
        lines.append(
            f"  {i}. {entry['class_name']:<45} {entry['confidence']*100:6.2f}%"
        )
    lines.append("")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict plant leaf disease from an image."
    )
    parser.add_argument("image", type=Path, help="Path to the leaf image file.")
    parser.add_argument("--top", type=int, default=3,
                        help="Number of top guesses to show (default: 3).")
    parser.add_argument("--model", type=Path, default=BEST_MODEL_PATH)
    args = parser.parse_args()

    fix_seeds(SEED)

    # Validate input file
    if not args.image.exists():
        print(f"[error] File not found: {args.image}")
        sys.exit(1)

    # Check that model exists
    if not args.model.exists():
        print(
            f"[error] Model not found: {args.model}\n"
            "Run 'python train.py' first to train and save the model."
        )
        sys.exit(1)

    try:
        model, classes, device = load_model(args.model)
        result = predict_image(args.image, model, classes, device, top_k=args.top)
        output = _format_result(result)
        try:
            print(output)
        except UnicodeEncodeError:
            print(output.encode("ascii", errors="replace").decode("ascii"))
    except ValueError as exc:
        print(f"[error] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
