"""Zero-shot CLIP gate that decides whether an image appears to be a plant leaf.

The gate uses the pretrained OpenAI CLIP ViT-B/32 model only.  It is never
trained or fine-tuned in this project.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from utils import get_device

MODEL_NAME = "openai/clip-vit-base-patch32"
CONFIG_PATH = Path("models/leaf_gate.json")
NOT_A_LEAF_MESSAGE = (
    "This does not look like a plant leaf. Please upload a clear photo of a "
    "single leaf in good light."
)
LEAF_PROMPTS = (
    "a close-up photo of a plant leaf",
    "a photo of a tomato leaf",
    "a photo of a diseased plant leaf",
    "a photo of a green leaf on a plant",
)
NON_LEAF_PROMPTS = (
    "a photo of a person",
    "a photo of an animal",
    "a photo of a building",
    "a photo of a vehicle",
    "a photo of food",
    "a photo of a flower",
    "a screenshot or a document",
    "a photo of a landscape",
    "a photo of an everyday object",
)


@dataclass
class LeafGate:
    """Lazy-loaded CLIP leaf gate; create one instance per process."""

    threshold: float | None = None
    device: torch.device | None = None
    model_name: str = MODEL_NAME
    _model: object | None = None
    _processor: object | None = None

    def load(self) -> "LeafGate":
        """Load CLIP once.  Do not substitute a model if weights are missing."""
        if self._model is not None:
            return self
        try:
            from transformers import CLIPModel, CLIPProcessor

            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model = CLIPModel.from_pretrained(self.model_name)
            self.device = self.device or get_device()
            self._model = self._model.to(self.device).eval()
        except Exception as exc:  # surface the exact download/load failure
            raise RuntimeError(
                f"Could not load CLIP weights for {self.model_name}: {exc}"
            ) from exc
        return self

    def load_threshold(self, path: Path = CONFIG_PATH) -> float:
        if self.threshold is not None:
            return self.threshold
        with path.open(encoding="utf-8") as handle:
            self.threshold = float(json.load(handle)["threshold"])
        return self.threshold

    def score(self, image: Image.Image) -> float:
        """Return total softmax probability assigned to all leaf prompts."""
        self.load()
        inputs = self._processor(
            text=[*LEAF_PROMPTS, *NON_LEAF_PROMPTS],
            images=image.convert("RGB"),
            return_tensors="pt",
            padding=True,
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits_per_image[0]
        return float(torch.softmax(logits, dim=0)[: len(LEAF_PROMPTS)].sum().item())

    def score_many(self, images: list[Image.Image]) -> list[float]:
        """Score a batch for offline tuning/evaluation without changing CLIP."""
        if not images:
            return []
        self.load()
        inputs = self._processor(
            text=[*LEAF_PROMPTS, *NON_LEAF_PROMPTS],
            images=[image.convert("RGB") for image in images],
            return_tensors="pt",
            padding=True,
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits_per_image
        probabilities = torch.softmax(logits, dim=1)[:, : len(LEAF_PROMPTS)].sum(dim=1)
        return [float(score) for score in probabilities.cpu().tolist()]

    def accepts(self, image: Image.Image) -> tuple[bool, float]:
        threshold = self.load_threshold()
        score = self.score(image)
        return score >= threshold, score


_gate: LeafGate | None = None


def get_leaf_gate() -> LeafGate:
    """Get the singleton used by the CLI and API process."""
    global _gate
    if _gate is None:
        _gate = LeafGate()
        _gate.load_threshold()
        _gate.load()
    return _gate
