"""SymbolClassifier — preprocessing + model wrapper for inference.

Loads its three artifact files at construction. If any are missing or the
model state_dict won't load, instance reports is_loaded() == False and
classify() returns None so callers fall through to Gemini.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import torch
    import torch.nn.functional as F
    from ml.model import SymbolCNN
    from ml.preprocessing import (
        binarize_and_count_components,
        crop_to_bbox,
        normalize,
        resize_to_32,
    )
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.85
MAX_COMPONENTS = 4


@dataclass
class ClassifyResult:
    predicted_latex: str
    confidence: float
    num_components: int
    accepted: bool


class SymbolClassifier:
    def __init__(
        self,
        model_path: Path,
        classes_path: Path,
        metrics_path: Path,
    ) -> None:
        self._enabled = False
        self._model = None
        self._classes: list[str] = []
        self._train_mean: float = 0.5
        self._train_std: float = 0.5
        if not _TORCH_AVAILABLE:
            logger.warning("Local classifier disabled (torch not installed).")
            return
        try:
            self._load(model_path, classes_path, metrics_path)
            self._enabled = True
        except FileNotFoundError:
            logger.warning(
                "Local classifier disabled (artifacts missing). "
                "Run `python -m ml.train` from backend/ to enable."
            )
        except Exception as exc:
            logger.error("Local classifier disabled: %s", exc)

    def _load(self, model_path: Path, classes_path: Path, metrics_path: Path) -> None:
        with open(classes_path, "r", encoding="utf-8") as f:
            self._classes = json.load(f)
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        self._train_mean = float(metrics["train_mean"])
        self._train_std = float(metrics["train_std"])
        self._model = SymbolCNN(num_classes=len(self._classes))
        state = torch.load(model_path, map_location="cpu")
        self._model.load_state_dict(state)
        self._model.eval()

    def is_loaded(self) -> bool:
        return self._enabled

    def _forward(self, tensor):
        """Override-able for testing."""
        assert self._model is not None
        with torch.no_grad():
            return self._model(tensor)

    def classify(self, image_bytes: bytes) -> Optional[ClassifyResult]:
        if not self._enabled:
            return None
        try:
            mask, num = binarize_and_count_components(image_bytes)
        except Exception as exc:
            logger.warning("Classifier preprocessing failed: %s", exc)
            return None
        if num == 0:
            return None
        if num > MAX_COMPONENTS:
            return ClassifyResult(
                predicted_latex="",
                confidence=0.0,
                num_components=num,
                accepted=False,
            )
        cropped = crop_to_bbox(mask, padding=4)
        arr = resize_to_32(cropped)
        tensor = normalize(arr, mean=self._train_mean, std=self._train_std)
        try:
            logits = self._forward(tensor)
            probs = F.softmax(logits, dim=1)
            confidence, idx = float(probs.max().item()), int(probs.argmax().item())
        except Exception as exc:
            logger.warning("Classifier inference failed: %s", exc)
            return None
        latex = self._classes[idx] if 0 <= idx < len(self._classes) else ""
        return ClassifyResult(
            predicted_latex=latex,
            confidence=confidence,
            num_components=num,
            accepted=confidence >= CONFIDENCE_THRESHOLD,
        )
