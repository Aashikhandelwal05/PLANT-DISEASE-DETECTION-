"""
api.py
======
FastAPI REST API for plant leaf disease prediction.

Endpoints
---------
  POST /api/v1/predict   Upload an image, get prediction JSON.
  GET  /health           Model status and class count.

Authentication
--------------
  All endpoints except /health require an X-API-Key header.
  Set API_KEY in a .env file (see README).

Error codes
-----------
  401  — Missing or wrong API key.
  415  — Uploaded file is not a recognised image type.
  422  — Validation error (e.g. no file in request).
  500  — Unexpected server error (no stack trace exposed).

Usage
-----
    uvicorn api:app --reload
    # then open http://localhost:8000/docs
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Security, UploadFile
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from PIL import Image, UnidentifiedImageError

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))
from predict import load_model, predict_image
from app.leaf_gate import LeafGate, get_leaf_gate
from utils import get_logger

load_dotenv()
logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Plant Disease Detection API",
    description=(
        "Upload a leaf image to detect plant diseases and receive treatment advice. "
        "Powered by EfficientNet-B0 fine-tuned on PlantVillage."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

ACCEPTED_CONTENT_TYPES = {
    "image/jpeg", "image/png", "image/bmp",
    "image/tiff", "image/webp",
}
ACCEPTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Load model once at startup
_model = None
_classes: list[str] = []
_device = None
_custom_model = None
_custom_classes: list[str] = []
_custom_device = None
_leaf_gate: LeafGate | None = None
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@app.on_event("startup")
def _startup() -> None:
    global _model, _classes, _device, _leaf_gate
    try:
        _model, _classes, _device = load_model()
        _leaf_gate = get_leaf_gate()
        logger.info("Model loaded. Classes: %d", len(_classes))
    except FileNotFoundError as exc:
        logger.error("Cannot start API — model not found: %s", exc)
        # Allow startup so /health can report the problem


# ──────────────────────────────────────────────────────────────────────────────
# Authentication
# ──────────────────────────────────────────────────────────────────────────────

_API_KEY: str = os.getenv("API_KEY", "")

if not _API_KEY:
    logger.error(
        "SECURITY WARNING: API_KEY is not set in .env — "
        "all prediction requests will be REJECTED with 401. "
        "Set API_KEY in your .env file to enable the API."
    )


def _verify_api_key(x_api_key: str | None = Security(_api_key_header)) -> None:
    """Dependency that checks the X-API-Key header."""
    if not _API_KEY:
        # Key not configured → lock everything, never allow through
        raise HTTPException(
            status_code=401,
            detail=(
                "API key not configured on server. "
                "Set API_KEY in the server's .env file."
            ),
        )
    if x_api_key is None or x_api_key != _API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Provide it in the X-API-Key header.",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Utility"])
def health() -> dict:
    """Return server and model status."""
    model_ready = _model is not None
    return {
        "status": "ok" if model_ready else "model_not_loaded",
        "model_ready": model_ready,
        "num_classes": len(_classes),
    }


@app.post(
    "/api/v1/predict",
    tags=["Prediction"],
    summary="Predict plant disease from leaf image",
    response_description="Prediction result with treatment advice.",
    dependencies=[Depends(_verify_api_key)],
)
async def predict(
    file: UploadFile = File(..., description="Leaf image (JPEG, PNG, BMP, TIFF, WebP)"),
) -> JSONResponse:
    """
    Upload a leaf image and receive:
    - **crop** and **disease** name
    - **confidence** score (0–1)
    - **confident** flag (True if confidence ≥ 90%)
    - **top_3** alternative predictions
    - **treatment** advice (symptoms, organic, chemical, prevention)
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs.",
        )

    # ── Validate file type ────────────────────────────────────────────────────
    content_type = (file.content_type or "").lower()
    file_ext = Path(file.filename or "").suffix.lower()

    if content_type not in ACCEPTED_CONTENT_TYPES and file_ext not in ACCEPTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported media type: '{content_type}'. "
                "Upload a JPEG, PNG, BMP, TIFF, or WebP image."
            ),
        )

    # ── Read and open image ───────────────────────────────────────────────────
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=415,
            detail="File could not be read as an image. Please upload a valid image.",
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to process uploaded file.")

    # ── Run inference ──────────────────────────────────────────────────────────
    # Save to a temp in-memory path so predict_image can open it
    tmp = io.BytesIO()
    img.save(tmp, format="PNG")
    tmp.seek(0)

    import tempfile, shutil

    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tf.write(tmp.read())
            tmp_path = Path(tf.name)

        result = predict_image(
            tmp_path, _model, _classes, _device, top_k=3, leaf_gate=_leaf_gate
        )
    except Exception as exc:
        logger.error("Inference failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Inference failed.")
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    # ── Build response ─────────────────────────────────────────────────────────
    if result.get("status") == "not_a_leaf":
        return JSONResponse(
            content={
                "status": "not_a_leaf",
                "message": result["message"],
                "leaf_score": round(result["leaf_score"], 4),
            }
        )

    return JSONResponse(
        content={
            "status": "ok",
            "leaf_score": round(result["leaf_score"], 4),
            "crop":       result["crop"],
            "disease":    result["disease"],
            "class_name": result["class_name"],
            "confidence": round(result["confidence"], 4),
            "top_3":      [
                {
                    "class_name": e["class_name"],
                    "confidence": round(e["confidence"], 4),
                }
                for e in result["top_k"]
            ],
            "treatment": result["treatment"],
        }
    )


def _get_custom_backend():
    """Load the CustomCNN once, without changing the configured default backend."""
    global _custom_model, _custom_classes, _custom_device
    if _custom_model is None:
        try:
            _custom_model, _custom_classes, _custom_device = load_model(backend="custom")
            logger.info("Custom CNN loaded. Classes: %d", len(_custom_classes))
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Custom CNN is unavailable: %s", exc)
            raise HTTPException(
                status_code=503,
                detail="Custom CNN not loaded. Run scripts/train_custom_cnn.py first.",
            ) from exc
    return _custom_model, _custom_classes, _custom_device


@app.post(
    "/api/v1/predict/custom",
    tags=["Prediction"],
    summary="Predict plant disease with the from-scratch Custom CNN",
    response_description="Prediction result with treatment advice.",
    dependencies=[Depends(_verify_api_key)],
)
async def predict_custom(
    file: UploadFile = File(..., description="Leaf image (JPEG, PNG, BMP, TIFF, WebP)"),
) -> JSONResponse:
    """Run the Custom CNN behind the same CLIP leaf gate as the default endpoint."""
    model, classes, device = _get_custom_backend()

    content_type = (file.content_type or "").lower()
    file_ext = Path(file.filename or "").suffix.lower()
    if content_type not in ACCEPTED_CONTENT_TYPES and file_ext not in ACCEPTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported media type: '{content_type}'. "
                "Upload a JPEG, PNG, BMP, TIFF, or WebP image."
            ),
        )

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=415,
            detail="File could not be read as an image. Please upload a valid image.",
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to process uploaded file.")

    tmp = io.BytesIO()
    img.save(tmp, format="PNG")
    tmp.seek(0)
    tmp_path: Path | None = None
    try:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tf.write(tmp.read())
            tmp_path = Path(tf.name)
        result = predict_image(
            tmp_path, model, classes, device, top_k=3, leaf_gate=_leaf_gate
        )
    except Exception as exc:
        logger.error("Custom CNN inference failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Inference failed.")
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    if result.get("status") == "not_a_leaf":
        return JSONResponse(
            content={
                "status": "not_a_leaf",
                "message": result["message"],
                "leaf_score": round(result["leaf_score"], 4),
            }
        )

    return JSONResponse(
        content={
            "status": "ok",
            "leaf_score": round(result["leaf_score"], 4),
            "crop": result["crop"],
            "disease": result["disease"],
            "class_name": result["class_name"],
            "confidence": round(result["confidence"], 4),
            "top_3": [
                {
                    "class_name": entry["class_name"],
                    "confidence": round(entry["confidence"], 4),
                }
                for entry in result["top_k"]
            ],
            "treatment": result["treatment"],
        }
    )
