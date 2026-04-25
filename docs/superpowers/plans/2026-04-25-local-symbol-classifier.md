# Local Symbol Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a server-side CNN classifier (trained on HASYv2 top-100 classes) that handles single-symbol drawings locally and falls back to Gemini for multi-symbol or low-confidence cases.

**Architecture:** New `backend/ml/` package containing model, preprocessing, training, and classifier modules. Classifier loads at FastAPI module import time and is consulted before Gemini in `/convert`. When `backend/ml/artifacts/` is empty (training not yet run), classifier reports `is_loaded() == False` and the request falls through to Gemini exactly as today — the app must keep working before training is done.

**Tech Stack:** PyTorch (CPU build), Pillow, numpy, FastAPI (existing), pytest (existing).

**Spec:** [`docs/superpowers/specs/2026-04-25-local-symbol-classifier-design.md`](../specs/2026-04-25-local-symbol-classifier-design.md)

---

## File Structure

```
backend/
  ml/
    __init__.py        — package marker
    model.py           — SymbolCNN nn.Module
    preprocessing.py   — shared crop/resize/normalize/component-count helpers
                         (used by BOTH train.py and classifier.py to keep them in sync)
    train.py           — `python -m ml.train` produces artifacts/
    classifier.py      — SymbolClassifier wraps preprocessing + model for inference
    README.md          — train / retrain / artifact docs
    artifacts/         — gitignored; produced by train.py
      model.pt
      classes.json
      metrics.json
  tests/
    test_classifier.py  — preprocessing, gating, disabled-state, mocked confidence
    test_train_smoke.py — 1-epoch synthetic-data training run
```

Modified:
- `backend/main.py` — startup classifier load, gating block, new response fields
- `backend/requirements.txt` — torch (CPU build) via `--extra-index-url`
- `backend/.gitignore` — `ml/artifacts/`
- `backend/tests/test_main.py` — three new tests for gating
- `frontend/mathboard/src/App.jsx` and `App.css` — source badge
- `SETUP.md` — short subsection on the local classifier

---

## Task 1: Bootstrap ml package + torch dependency

**Files:**
- Create: `backend/ml/__init__.py`
- Create: `backend/.gitignore`
- Modify: `backend/requirements.txt`

- [ ] **Step 1: Create `backend/.gitignore`**

Write file `backend/.gitignore`:
```
ml/artifacts/
__pycache__/
*.pyc
.venv/
.env
.pytest_cache/
```

- [ ] **Step 2: Create empty package marker `backend/ml/__init__.py`**

Write file `backend/ml/__init__.py` with content:
```python
"""Local symbol classifier package — see backend/ml/README.md."""
```

- [ ] **Step 3: Add torch (CPU build) to `backend/requirements.txt`**

Replace the contents of `backend/requirements.txt` with:
```
--extra-index-url https://download.pytorch.org/whl/cpu
fastapi
uvicorn
python-multipart
google-genai
python-dotenv
sympy
Pillow
pytest
httpx
torch
```

The `--extra-index-url` line tells pip to fetch the CPU-only torch build (~200 MB) from the PyTorch index, falling back to PyPI for everything else. Without that line, plain `pip install torch` on a Windows machine pulls the CUDA build (~2 GB).

- [ ] **Step 4: Install torch in the existing venv**

Run from `backend/` (PowerShell or bash):
```
.venv/Scripts/python -m pip install -r requirements.txt
```

Expected: pip downloads `torch` from `download.pytorch.org/whl/cpu`. ~1-3 minutes. No errors.

- [ ] **Step 5: Smoke-verify torch imports**

Run from `backend/`:
```
.venv/Scripts/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected output:
```
2.x.x False
```
The `False` is correct — we explicitly want the CPU build.

- [ ] **Step 6: Commit**

```
git add backend/.gitignore backend/ml/__init__.py backend/requirements.txt
git commit -m "chore: scaffold ml package, add torch (CPU) dependency"
```

---

## Task 2: SymbolCNN model class

**Files:**
- Create: `backend/ml/model.py`
- Create: `backend/tests/test_classifier.py` (with first test)

- [ ] **Step 1: Write the failing test** in `backend/tests/test_classifier.py`

Create the file with:
```python
import torch

from ml.model import SymbolCNN


def test_symbol_cnn_forward_shape():
    """Forward pass returns logits of shape (batch, num_classes)."""
    model = SymbolCNN(num_classes=100)
    x = torch.zeros(2, 1, 32, 32)
    logits = model(x)
    assert logits.shape == (2, 100)


def test_symbol_cnn_param_count_under_500k():
    """Model is intentionally tiny so CPU inference and training are fast."""
    model = SymbolCNN(num_classes=100)
    n = sum(p.numel() for p in model.parameters())
    assert n < 500_000, f"SymbolCNN has {n:,} params (expected <500k)"
```

- [ ] **Step 2: Run the tests, expect failure**

Run from `backend/`:
```
.venv/Scripts/python -m pytest tests/test_classifier.py -v
```

Expected: collection error or `ModuleNotFoundError: No module named 'ml.model'`.

- [ ] **Step 3: Implement `backend/ml/model.py`**

Create the file:
```python
"""SymbolCNN — small CNN for 32x32 grayscale symbol classification."""

import torch.nn as nn


class SymbolCNN(nn.Module):
    """Five-layer convnet, ~250k params. Designed for fast CPU training/inference."""

    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

- [ ] **Step 4: Run the tests, expect pass**

Run from `backend/`:
```
.venv/Scripts/python -m pytest tests/test_classifier.py -v
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```
git add backend/ml/model.py backend/tests/test_classifier.py
git commit -m "feat(ml): add SymbolCNN model class with shape + param-count tests"
```

---

## Task 3: Shared preprocessing utilities

**Files:**
- Create: `backend/ml/preprocessing.py`
- Modify: `backend/tests/test_classifier.py` (append tests)

- [ ] **Step 1: Append failing tests** to `backend/tests/test_classifier.py`

Add to the bottom of the file:
```python
import io
import numpy as np
from PIL import Image, ImageDraw

from ml.preprocessing import (
    binarize_and_count_components,
    crop_to_bbox,
    resize_to_32,
    normalize,
)


def _make_image(width: int, height: int, draw_fn) -> bytes:
    img = Image.new("L", (width, height), color=255)
    draw_fn(ImageDraw.Draw(img))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_binarize_counts_one_component_for_single_blob():
    img_bytes = _make_image(200, 80, lambda d: d.ellipse((20, 20, 60, 60), fill=0))
    mask, num = binarize_and_count_components(img_bytes)
    assert num == 1
    assert mask.dtype == bool
    assert mask.shape == (80, 200)


def test_binarize_counts_five_components_for_five_dots():
    def draw(d):
        for cx in (20, 50, 80, 110, 140):
            d.ellipse((cx, 30, cx + 10, 40), fill=0)
    img_bytes = _make_image(200, 80, draw)
    _, num = binarize_and_count_components(img_bytes)
    assert num == 5


def test_crop_resize_normalize_produces_32x32_tensor():
    img_bytes = _make_image(200, 80, lambda d: d.ellipse((150, 50, 180, 70), fill=0))
    mask, _ = binarize_and_count_components(img_bytes)
    cropped = crop_to_bbox(mask, padding=4)
    resized = resize_to_32(cropped)
    arr = normalize(resized, mean=0.5, std=0.5)
    assert arr.shape == (1, 1, 32, 32)
    # ink centered (mass roughly in the middle 16x16)
    middle_mass = float(arr[0, 0, 8:24, 8:24].abs().sum())
    edge_mass = float(arr[0, 0].abs().sum() - middle_mass)
    assert middle_mass > edge_mass
```

- [ ] **Step 2: Run tests, expect failure**

```
.venv/Scripts/python -m pytest tests/test_classifier.py -v
```

Expected: `ImportError` for `ml.preprocessing`.

- [ ] **Step 3: Implement `backend/ml/preprocessing.py`**

Create the file:
```python
"""Shared preprocessing for both training and inference.

Both pipelines MUST use these functions to keep training and serving in sync.
"""

import io
from typing import Tuple

import numpy as np
import torch
from PIL import Image


def _otsu_threshold(gray: np.ndarray) -> int:
    """Standard Otsu threshold on a uint8 grayscale array."""
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    total = gray.size
    sum_total = float((np.arange(256) * hist).sum())
    sum_b, w_b, max_var, threshold = 0.0, 0, 0.0, 0
    for t in range(256):
        w_b += int(hist[t])
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += float(t * hist[t])
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > max_var:
            max_var, threshold = var, t
    return threshold


def binarize_and_count_components(image_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Decode PNG -> grayscale -> binary ink mask + connected component count.

    Returns (mask, num_components) where mask is bool array, True = ink pixel.
    Caller may early-return on num_components > 4 (multi-symbol) or 0 (blank).
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    arr = np.asarray(img, dtype=np.uint8)
    threshold = _otsu_threshold(arr)
    mask = arr < threshold  # ink = darker than threshold
    num = _count_components(mask)
    return mask, num


def _count_components(mask: np.ndarray) -> int:
    """4-connectivity flood-fill component count. No scipy dep."""
    visited = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    count = 0
    stack: list[tuple[int, int]] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            count += 1
            stack.append((y, x))
            while stack:
                cy, cx = stack.pop()
                if cy < 0 or cy >= h or cx < 0 or cx >= w:
                    continue
                if visited[cy, cx] or not mask[cy, cx]:
                    continue
                visited[cy, cx] = True
                stack.extend(((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)))
    return count


def crop_to_bbox(mask: np.ndarray, padding: int = 4) -> np.ndarray:
    """Crop a binary mask to its ink bounding box plus padding pixels."""
    if not mask.any():
        return mask
    ys, xs = np.where(mask)
    y0, y1 = max(0, ys.min() - padding), min(mask.shape[0], ys.max() + padding + 1)
    x0, x1 = max(0, xs.min() - padding), min(mask.shape[1], xs.max() + padding + 1)
    return mask[y0:y1, x0:x1]


def resize_to_32(mask: np.ndarray) -> np.ndarray:
    """Resize a (H,W) bool/float mask to 32x32 using LANCZOS, ink white on black.

    HASYv2 is white-ink-on-black. We invert here so the model sees the same
    polarity at training and inference time.
    """
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    img = img.resize((32, 32), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def normalize(arr_32x32: np.ndarray, mean: float, std: float) -> torch.Tensor:
    """Normalize a (32,32) float array to a (1,1,32,32) tensor."""
    t = torch.from_numpy(arr_32x32).float().unsqueeze(0).unsqueeze(0)
    return (t - mean) / std
```

- [ ] **Step 4: Run tests, expect pass**

```
.venv/Scripts/python -m pytest tests/test_classifier.py -v
```

Expected: 5 passed (2 model + 3 preprocessing).

- [ ] **Step 5: Commit**

```
git add backend/ml/preprocessing.py backend/tests/test_classifier.py
git commit -m "feat(ml): add shared preprocessing (otsu, components, crop, resize, normalize)"
```

---

## Task 4: SymbolClassifier — full implementation

**Files:**
- Create: `backend/ml/classifier.py`
- Modify: `backend/tests/test_classifier.py` (append tests)

- [ ] **Step 1: Append failing tests** to `backend/tests/test_classifier.py`

Add to the bottom of the file:
```python
from pathlib import Path
from unittest.mock import patch

from ml.classifier import SymbolClassifier


def test_classifier_disabled_when_artifacts_missing(tmp_path):
    c = SymbolClassifier(
        model_path=tmp_path / "missing.pt",
        classes_path=tmp_path / "missing.json",
        metrics_path=tmp_path / "missing.json",
    )
    assert c.is_loaded() is False
    img = _make_image(50, 50, lambda d: d.rectangle((10, 10, 30, 30), fill=0))
    assert c.classify(img) is None


def test_classifier_skips_when_too_many_components(tmp_path):
    """Five disjoint dots -> classifier should not run model, returns accepted=False."""
    c = SymbolClassifier(
        model_path=tmp_path / "missing.pt",
        classes_path=tmp_path / "missing.json",
        metrics_path=tmp_path / "missing.json",
    )
    # Force enabled state so we exercise the gate, not the disabled path.
    c._enabled = True
    c._classes = ["x"] * 100
    c._train_mean = 0.5
    c._train_std = 0.5

    def draw(d):
        for cx in (5, 25, 45, 65, 85):
            d.ellipse((cx, 30, cx + 5, 35), fill=0)
    img = _make_image(100, 60, draw)
    result = c.classify(img)
    assert result is not None
    assert result.accepted is False
    assert result.predicted_latex == ""
    assert result.num_components == 5


def test_classifier_accepts_high_confidence(tmp_path):
    c = SymbolClassifier(
        model_path=tmp_path / "missing.pt",
        classes_path=tmp_path / "missing.json",
        metrics_path=tmp_path / "missing.json",
    )
    c._enabled = True
    c._classes = ["x", "y"] + ["z"] * 98
    c._train_mean = 0.5
    c._train_std = 0.5

    fake_logits = torch.tensor([[10.0, 0.0] + [0.0] * 98])  # softmax ~0.9999 on idx 0
    with patch.object(c, "_forward", return_value=fake_logits):
        img = _make_image(60, 60, lambda d: d.ellipse((20, 20, 40, 40), fill=0))
        result = c.classify(img)
    assert result is not None
    assert result.accepted is True
    assert result.predicted_latex == "x"
    assert result.confidence > 0.9


def test_classifier_rejects_low_confidence(tmp_path):
    c = SymbolClassifier(
        model_path=tmp_path / "missing.pt",
        classes_path=tmp_path / "missing.json",
        metrics_path=tmp_path / "missing.json",
    )
    c._enabled = True
    c._classes = ["x", "y"] + ["z"] * 98
    c._train_mean = 0.5
    c._train_std = 0.5

    # Near-uniform logits -> top-1 prob ~0.5
    fake_logits = torch.tensor([[0.5, 0.4] + [0.0] * 98])
    with patch.object(c, "_forward", return_value=fake_logits):
        img = _make_image(60, 60, lambda d: d.ellipse((20, 20, 40, 40), fill=0))
        result = c.classify(img)
    assert result is not None
    assert result.accepted is False
    assert result.predicted_latex == "x"  # top-1 still reported
    assert result.confidence < 0.85


def test_classifier_returns_none_for_blank_image(tmp_path):
    c = SymbolClassifier(
        model_path=tmp_path / "missing.pt",
        classes_path=tmp_path / "missing.json",
        metrics_path=tmp_path / "missing.json",
    )
    c._enabled = True
    c._classes = ["x"] * 100
    c._train_mean = 0.5
    c._train_std = 0.5
    blank = _make_image(50, 50, lambda d: None)
    assert c.classify(blank) is None
```

- [ ] **Step 2: Run tests, expect failure**

```
.venv/Scripts/python -m pytest tests/test_classifier.py -v
```

Expected: ImportError for `ml.classifier`.

- [ ] **Step 3: Implement `backend/ml/classifier.py`**

Create the file:
```python
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

import torch
import torch.nn.functional as F

from ml.model import SymbolCNN
from ml.preprocessing import (
    binarize_and_count_components,
    crop_to_bbox,
    normalize,
    resize_to_32,
)

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
        self._model: Optional[SymbolCNN] = None
        self._classes: list[str] = []
        self._train_mean: float = 0.5
        self._train_std: float = 0.5
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

    def _forward(self, tensor: torch.Tensor) -> torch.Tensor:
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
```

- [ ] **Step 4: Run tests, expect pass**

```
.venv/Scripts/python -m pytest tests/test_classifier.py -v
```

Expected: 10 passed (2 model + 3 preprocessing + 5 classifier).

- [ ] **Step 5: Commit**

```
git add backend/ml/classifier.py backend/tests/test_classifier.py
git commit -m "feat(ml): add SymbolClassifier with disabled/gating/inference paths + tests"
```

---

## Task 5: Training script + smoke test

**Files:**
- Create: `backend/ml/train.py`
- Create: `backend/tests/test_train_smoke.py`

- [ ] **Step 1: Write the failing smoke test** at `backend/tests/test_train_smoke.py`

Create the file:
```python
"""Smoke test for ml/train.py — runs 1 epoch on a synthetic dataset."""

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

import ml.train as train_module


def _make_synthetic_dataset(root: Path, classes: list[str], samples_per_class: int) -> Path:
    images_dir = root / "hasy-data"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_path = root / "hasy-data-labels.csv"
    rows = []
    rng = np.random.default_rng(0)
    idx = 0
    for cls in classes:
        for _ in range(samples_per_class):
            arr = (rng.random((32, 32)) * 255).astype(np.uint8)
            fname = f"v2-{idx:05d}.png"
            Image.fromarray(arr, mode="L").save(images_dir / fname)
            rows.append({
                "path": f"hasy-data/{fname}",
                "symbol_id": str(idx),
                "latex": cls,
                "user_id": "0",
            })
            idx += 1
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "symbol_id", "latex", "user_id"])
        w.writeheader()
        w.writerows(rows)
    return root


def test_training_smoke(tmp_path):
    classes = ["A", "B", "C", "D", "E"]
    data_root = _make_synthetic_dataset(tmp_path / "data", classes, samples_per_class=40)
    artifacts_dir = tmp_path / "artifacts"

    train_module.run(
        data_root=data_root,
        artifacts_dir=artifacts_dir,
        top_n_classes=5,
        epochs=1,
        batch_size=16,
        seed=42,
    )

    assert (artifacts_dir / "model.pt").exists()
    assert (artifacts_dir / "classes.json").exists()
    assert (artifacts_dir / "metrics.json").exists()

    with open(artifacts_dir / "classes.json", encoding="utf-8") as f:
        saved_classes = json.load(f)
    assert sorted(saved_classes) == sorted(classes)

    with open(artifacts_dir / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    for key in ("val_top1_acc", "coverage_at_threshold", "accuracy_on_accepted",
                "train_mean", "train_std"):
        assert key in metrics, f"missing key in metrics.json: {key}"
```

- [ ] **Step 2: Run smoke test, expect failure**

```
.venv/Scripts/python -m pytest tests/test_train_smoke.py -v
```

Expected: ImportError for `ml.train`.

- [ ] **Step 3: Implement `backend/ml/train.py`**

Create the file:
```python
"""Train a SymbolCNN on HASYv2 top-N classes.

Run from backend/:
    python -m ml.train

Reads data/hasy/hasy-data-labels.csv (+ data/hasy/hasy-data/*.png),
writes ml/artifacts/{model.pt, classes.json, metrics.json}.
"""

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ml.model import SymbolCNN

CONFIDENCE_THRESHOLD = 0.85
DEFAULT_TOP_N = 100
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_SEED = 42


class HasyDataset(Dataset):
    def __init__(
        self,
        rows: list[tuple[str, int]],
        image_root: Path,
        augment: bool,
        mean: float,
        std: float,
    ) -> None:
        self.rows = rows
        self.image_root = image_root
        self.augment = augment
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, rel_path: str) -> np.ndarray:
        with Image.open(self.image_root / rel_path) as img:
            arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        if arr.shape != (32, 32):
            # HASYv2 is 32x32, but be defensive.
            img2 = Image.fromarray((arr * 255).astype(np.uint8), mode="L").resize(
                (32, 32), Image.LANCZOS
            )
            arr = np.asarray(img2, dtype=np.float32) / 255.0
        return arr

    def _augment(self, arr: np.ndarray) -> np.ndarray:
        # Random rotation +/- 10 deg, random translation +/- 2 px.
        img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
        angle = random.uniform(-10, 10)
        tx, ty = random.randint(-2, 2), random.randint(-2, 2)
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=255)
        img = img.transform(
            img.size, Image.AFFINE, (1, 0, tx, 0, 1, ty), fillcolor=255
        )
        return np.asarray(img, dtype=np.float32) / 255.0

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        rel, label = self.rows[i]
        arr = self._load(rel)
        if self.augment:
            arr = self._augment(arr)
        t = torch.from_numpy(arr).unsqueeze(0)
        t = (t - self.mean) / self.std
        return t, label


def _read_labels(labels_csv: Path) -> list[dict]:
    with open(labels_csv, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pick_top_classes(rows: list[dict], top_n: int) -> list[str]:
    counts = Counter(r["latex"] for r in rows)
    # frequency desc, latex asc on ties
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in ordered[:top_n]]


def _stratified_split(
    rows: list[tuple[str, int]], val_frac: float, seed: int
) -> tuple[list, list]:
    rng = random.Random(seed)
    by_label: dict[int, list[tuple[str, int]]] = {}
    for r in rows:
        by_label.setdefault(r[1], []).append(r)
    train, val = [], []
    for label, items in by_label.items():
        rng.shuffle(items)
        cut = max(1, int(len(items) * val_frac))
        val.extend(items[:cut])
        train.extend(items[cut:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _compute_dataset_stats(rows: list[tuple[str, int]], image_root: Path) -> tuple[float, float]:
    """Mean and std over a sample of training images (max 2000)."""
    sample = rows[: min(2000, len(rows))]
    pixels: list[np.ndarray] = []
    for rel, _ in sample:
        with Image.open(image_root / rel) as img:
            arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        pixels.append(arr.flatten())
    flat = np.concatenate(pixels)
    return float(flat.mean()), float(flat.std() or 1e-6)


def _train_one_epoch(
    model: SymbolCNN, loader: DataLoader, optim, criterion, device
) -> tuple[float, float]:
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optim.step()
        total_loss += float(loss.item()) * x.size(0)
        correct += int((logits.argmax(1) == y).sum().item())
        n += x.size(0)
    return total_loss / max(1, n), correct / max(1, n)


@torch.no_grad()
def _evaluate(
    model: SymbolCNN, loader: DataLoader, criterion, device
) -> tuple[float, float, float, float]:
    """Returns (val_loss, top1_acc, coverage_at_threshold, acc_on_accepted)."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    accepted, accepted_correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        total_loss += float(loss.item()) * x.size(0)
        correct += int((pred == y).sum().item())
        n += x.size(0)
        mask = conf >= CONFIDENCE_THRESHOLD
        accepted += int(mask.sum().item())
        accepted_correct += int(((pred == y) & mask).sum().item())
    return (
        total_loss / max(1, n),
        correct / max(1, n),
        accepted / max(1, n),
        accepted_correct / max(1, accepted),
    )


def run(
    data_root: Path,
    artifacts_dir: Path,
    top_n_classes: int = DEFAULT_TOP_N,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
) -> None:
    """Train a SymbolCNN. Used by CLI and by smoke tests with patched paths."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    labels_csv = data_root / "hasy-data-labels.csv"
    image_root = data_root

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"[train] Reading labels from {labels_csv}", flush=True)
    raw = _read_labels(labels_csv)
    print(f"[train] {len(raw):,} rows", flush=True)

    classes = _pick_top_classes(raw, top_n_classes)
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    rows = [(r["path"], cls_to_idx[r["latex"]]) for r in raw if r["latex"] in cls_to_idx]
    print(
        f"[train] Using top {len(classes)} classes, {len(rows):,} samples "
        f"({len(rows)/max(1,len(raw)):.1%} of total)",
        flush=True,
    )

    train_rows, val_rows = _stratified_split(rows, val_frac=0.15, seed=seed)
    print(f"[train] split: {len(train_rows):,} train / {len(val_rows):,} val", flush=True)

    mean, std = _compute_dataset_stats(train_rows, image_root)
    print(f"[train] dataset mean={mean:.4f} std={std:.4f}", flush=True)

    train_ds = HasyDataset(train_rows, image_root, augment=True, mean=mean, std=std)
    val_ds = HasyDataset(val_rows, image_root, augment=False, mean=mean, std=std)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SymbolCNN(num_classes=len(classes)).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, epochs))
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_state = None
    best_metrics = {}
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _train_one_epoch(model, train_loader, optim, criterion, device)
        val_loss, val_acc, coverage, acc_accepted = _evaluate(model, val_loader, criterion, device)
        scheduler.step()
        print(
            f"[train] epoch {epoch}/{epochs} "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"coverage@{CONFIDENCE_THRESHOLD:.2f}={coverage:.4f} "
            f"acc_on_accepted={acc_accepted:.4f}",
            flush=True,
        )
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_metrics = {
                "val_top1_acc": val_acc,
                "coverage_at_threshold": coverage,
                "accuracy_on_accepted": acc_accepted,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "train_mean": mean,
                "train_std": std,
                "epoch": epoch,
                "num_classes": len(classes),
            }

    torch.save(best_state, artifacts_dir / "model.pt")
    with open(artifacts_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
    with open(artifacts_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)
    print(
        f"[train] Best val_acc={best_acc:.4f}. Wrote artifacts to {artifacts_dir}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/hasy", type=Path)
    parser.add_argument("--artifacts-dir", default="ml/artifacts", type=Path)
    parser.add_argument("--top-n", default=DEFAULT_TOP_N, type=int)
    parser.add_argument("--epochs", default=DEFAULT_EPOCHS, type=int)
    parser.add_argument("--batch-size", default=DEFAULT_BATCH_SIZE, type=int)
    parser.add_argument("--seed", default=DEFAULT_SEED, type=int)
    args = parser.parse_args()
    run(
        data_root=args.data_root,
        artifacts_dir=args.artifacts_dir,
        top_n_classes=args.top_n,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run smoke test, expect pass**

```
.venv/Scripts/python -m pytest tests/test_train_smoke.py -v
```

Expected: 1 passed in ~10-30 seconds.

- [ ] **Step 5: Run full test suite to confirm nothing else broke**

```
.venv/Scripts/python -m pytest tests/ -v
```

Expected: all tests pass (existing 31 + 11 new = 42 total).

- [ ] **Step 6: Commit**

```
git add backend/ml/train.py backend/tests/test_train_smoke.py
git commit -m "feat(ml): training script + smoke test"
```

---

## Task 6: Run real training on HASYv2

This task is a manual run, not TDD. It produces the artifacts the classifier loads at runtime.

**Files:**
- Produces: `backend/ml/artifacts/{model.pt, classes.json, metrics.json}` (gitignored)

- [ ] **Step 1: Verify HASYv2 is downloaded**

Run from `backend/`:
```
ls data/hasy/hasy-data | head -3 && ls data/hasy/hasy-data-labels.csv
```

Expected: at least 3 PNG filenames + the labels CSV. If the directory is empty, see SETUP.md step 5 (HASYv2 was downloaded earlier in the project — should be present).

- [ ] **Step 2: Train**

Run from `backend/`:
```
.venv/Scripts/python -m ml.train
```

Expected: ~30 minutes on CPU. Per-epoch lines like:
```
[train] epoch 5/10 train_loss=0.42 train_acc=0.89 val_loss=0.55 val_acc=0.84 coverage@0.85=0.71 acc_on_accepted=0.95
```
Final line:
```
[train] Best val_acc=0.85. Wrote artifacts to ml/artifacts
```

- [ ] **Step 3: Sanity-check the artifacts**

Run from `backend/`:
```
.venv/Scripts/python -c "import json; print(json.dumps(json.load(open('ml/artifacts/metrics.json')), indent=2))"
.venv/Scripts/python -c "import json; print(len(json.load(open('ml/artifacts/classes.json'))), 'classes')"
ls -lh ml/artifacts/model.pt
```

Expected:
- `metrics.json` has `val_top1_acc` ≥ ~0.75 and `accuracy_on_accepted` ≥ ~0.90.
- `classes.json` has 100 entries.
- `model.pt` is roughly 1-3 MB.

If `accuracy_on_accepted` is below 0.85, raise `CONFIDENCE_THRESHOLD` in `classifier.py` and `train.py` to tighten acceptance — the threshold sweep printed during training tells you what value gives the right tradeoff.

- [ ] **Step 4: No commit (artifacts are gitignored)**

Verify with:
```
git status
```
Expected: working tree clean.

---

## Task 7: Wire classifier into /convert + extend response

**Files:**
- Modify: `backend/main.py`
- Modify: `backend/tests/test_main.py`

- [ ] **Step 1: Append three new failing tests** to `backend/tests/test_main.py`

Add to the bottom of the file:
```python
from ml.classifier import ClassifyResult


def test_convert_uses_local_when_classifier_accepts():
    """When the local classifier accepts, Gemini is NOT called and source='local'."""
    accepted = ClassifyResult(
        predicted_latex="x", confidence=0.93, num_components=1, accepted=True
    )

    class FakeClassifier:
        def is_loaded(self): return True
        def classify(self, _): return accepted

    with patch.object(main, "classifier", FakeClassifier()), \
         patch.object(main._client.models, "generate_content") as gemini, \
         patch.object(main.solver, "solve_expression", return_value=_MOCK_SOLVER_RESULT):
        resp = _client.post("/convert", files={"file": ("b.png", _png_bytes(), "image/png")})

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["latex"] == "x"
    assert data["source"] == "local"
    assert 0.9 < data["confidence"] < 1.0
    gemini.assert_not_called()


def test_convert_falls_back_to_gemini_on_low_confidence():
    rejected = ClassifyResult(
        predicted_latex="x", confidence=0.40, num_components=1, accepted=False
    )

    class FakeClassifier:
        def is_loaded(self): return True
        def classify(self, _): return rejected

    mock_response = MagicMock()
    mock_response.text = "x^2 + 1"

    with patch.object(main, "classifier", FakeClassifier()), \
         patch.object(main._client.models, "generate_content", return_value=mock_response), \
         patch.object(main.solver, "solve_expression", return_value=_MOCK_SOLVER_RESULT):
        resp = _client.post("/convert", files={"file": ("b.png", _png_bytes(), "image/png")})

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["latex"] == "x^2 + 1"
    assert data["source"] == "gemini"
    assert data["confidence"] is None


def test_convert_falls_back_when_classifier_disabled():
    class DisabledClassifier:
        def is_loaded(self): return False
        def classify(self, _): return None

    mock_response = MagicMock()
    mock_response.text = "y"

    with patch.object(main, "classifier", DisabledClassifier()), \
         patch.object(main._client.models, "generate_content", return_value=mock_response), \
         patch.object(main.solver, "solve_expression", return_value=_MOCK_SOLVER_RESULT):
        resp = _client.post("/convert", files={"file": ("b.png", _png_bytes(), "image/png")})

    assert resp.json()["source"] == "gemini"
    assert resp.json()["confidence"] is None
```

- [ ] **Step 2: Run tests, expect failure**

```
.venv/Scripts/python -m pytest tests/test_main.py -v
```

Expected: the three new tests fail (no `classifier` attribute on `main`, no `source` in response). Existing tests still pass.

- [ ] **Step 3: Modify `backend/main.py`**

Replace the existing file with the version below. Key changes vs current file:
- Restore `from google.genai import types` (existing code at line 125 needs it).
- Restore `from dotenv import load_dotenv` and call it before reading env vars (otherwise `.env` isn't picked up when running `uvicorn main:app` directly).
- Add `from pathlib import Path`.
- Add `from ml.classifier import SymbolClassifier` and instantiate as a module-level `classifier`.
- Insert gating block in `/convert` before the Gemini call.
- Add `source` and `confidence` to the response.

```python
"""
MathBoard FastAPI backend.

Accepts a handwritten-math image. Tries a local SymbolClassifier first; if it
declines (artifacts missing, multi-symbol, or low confidence), falls back to
Google Gemini for OCR. The result (LaTeX) goes through solver.py (SymPy).
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

import solver
from ml.classifier import SymbolClassifier

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

if not GEMINI_API_KEY:
    print(
        "\n"
        "================================================================\n"
        "  WARNING: GEMINI_API_KEY is not set.\n"
        "  Create backend/.env with GEMINI_API_KEY=your_key_here\n"
        "  Generate a key at https://aistudio.google.com\n"
        "================================================================\n",
        file=sys.stderr,
        flush=True,
    )
    _client = None
else:
    _client = genai.Client(api_key=GEMINI_API_KEY)

_ARTIFACTS = Path(__file__).parent / "ml" / "artifacts"
classifier = SymbolClassifier(
    model_path=_ARTIFACTS / "model.pt",
    classes_path=_ARTIFACTS / "classes.json",
    metrics_path=_ARTIFACTS / "metrics.json",
)
print(
    f"[Classifier] {'enabled' if classifier.is_loaded() else 'disabled (run python -m ml.train to enable)'}",
    flush=True,
)

OCR_PROMPT = (
    "You are an expert mathematical OCR system. "
    "Extract the mathematical expression from this image and return ONLY the "
    "raw LaTeX string. Do not include any markdown formatting, backticks, or "
    "conversational text. Do not include \\( or \\) wrappers. Just the math."
)

app = FastAPI(title="MathBoard Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _strip_latex_fences(latex: str) -> str:
    if not latex:
        return latex
    s = latex.strip()
    if s.startswith("```"):
        first_newline = s.find("\n")
        s = s[first_newline + 1:] if first_newline != -1 else s.lstrip("`")
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    s = s.strip("`").strip()
    if s.startswith("\\(") and s.endswith("\\)"):
        s = s[2:-2].strip()
    if s.startswith("$$") and s.endswith("$$"):
        s = s[2:-2].strip()
    elif s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    return s


def _classify_gemini_error(exc: Exception) -> str:
    msg = str(exc)
    lowered = msg.lower()
    if "service_disabled" in lowered or "has not been used in project" in lowered:
        import re as _re
        m = _re.search(r"https://console\.developers\.google\.com/[^\s'\"]+", msg)
        url = m.group(0) if m else "https://aistudio.google.com"
        return (
            "Your Gemini key is a Google Cloud key whose project does not have the "
            "Generative Language API enabled. Either (a) enable it here: "
            f"{url} (then wait ~2 min), or (b) replace the key in backend/.env "
            "with an AI Studio key from https://aistudio.google.com (works immediately)."
        )
    if "403" in msg or "permission_denied" in lowered:
        return "Gemini API key invalid or unauthorized. Generate a new AI Studio key at https://aistudio.google.com."
    if "401" in msg or "unauthenticated" in lowered or "api key not valid" in lowered:
        return "Gemini API key is not authorized. Generate a fresh key at https://aistudio.google.com."
    if "429" in msg or "quota" in lowered or "rate" in lowered:
        return "Gemini quota / rate limit exceeded. Wait a bit and try again."
    if "deadline" in lowered or "timeout" in lowered:
        return "Gemini request timed out. Check your internet connection and retry."
    return f"Gemini API error: {msg}"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "gemini_key_set": bool(GEMINI_API_KEY),
        "gemini_model": GEMINI_MODEL,
        "classifier_loaded": classifier.is_loaded(),
    }


@app.post("/convert")
async def convert_equation(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
    except Exception as exc:
        return {"error": f"Failed to read uploaded file: {exc}"}

    if not image_bytes:
        return {"error": "Uploaded image was empty."}

    # Try local classifier first.
    try:
        local_result = classifier.classify(image_bytes) if classifier.is_loaded() else None
    except Exception as exc:
        print(f"[Classifier Error] {exc}", flush=True)
        local_result = None

    if local_result and local_result.accepted:
        latex_string = local_result.predicted_latex
        source = "local"
        confidence: float | None = local_result.confidence
        print(f"[Classifier] Accepted '{latex_string}' (conf={confidence:.3f})", flush=True)
    else:
        if not GEMINI_API_KEY or _client is None:
            return {
                "error": (
                    "Gemini API key is not configured on the server. "
                    "Add GEMINI_API_KEY=... to backend/.env and restart."
                )
            }
        print("[Routing] Sending image to Gemini...", flush=True)
        try:
            response = _client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    OCR_PROMPT,
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                ],
            )
            raw_latex = (response.text or "").strip()
        except Exception as exc:
            print(f"[Gemini Error] {exc}", flush=True)
            return {"error": _classify_gemini_error(exc)}
        latex_string = _strip_latex_fences(raw_latex)
        if not latex_string:
            return {
                "error": (
                    "Gemini returned an empty response. Try drawing a clearer "
                    "expression or using a thicker pen."
                )
            }
        source = "gemini"
        confidence = None
        print(f"[Gemini Result] {latex_string}", flush=True)

    solution_data = solver.solve_expression(latex_string)
    print(f"[Solver Result] {solution_data}", flush=True)

    return {
        "status": "success",
        "latex": latex_string,
        "solution_data": solution_data,
        "source": source,
        "confidence": confidence,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

- [ ] **Step 4: Run all tests, expect pass**

```
.venv/Scripts/python -m pytest tests/ -v
```

Expected: 45 passed (existing 31 + 11 classifier + 1 train smoke + 3 convert gating). The previous `test_convert_happy_path` keeps passing because the classifier is in disabled state during tests (no artifacts) and the Gemini path is exercised exactly as before — but the response now also has `source="gemini"`/`confidence=None`, which existing tests don't assert against.

- [ ] **Step 5: Manual smoke test against the running backend**

If a backend is already running, restart it so the new `main.py` is loaded.
Then run from `backend/`:
```
.venv/Scripts/python -c "
import io, requests
from PIL import Image, ImageDraw, ImageFont
img = Image.new('RGB', (300, 100), 'white')
d = ImageDraw.Draw(img)
try: f = ImageFont.truetype('arial.ttf', 56)
except Exception: f = ImageFont.load_default()
d.text((30, 20), 'x = 5', fill='black', font=f)
buf = io.BytesIO(); img.save(buf, format='PNG')
import json
r = requests.post('http://127.0.0.1:8000/convert', files={'file': ('t.png', buf.getvalue(), 'image/png')}, timeout=30)
print(json.dumps(r.json(), indent=2))
"
```

Expected: a JSON response that includes `"source": "local"` (if the classifier accepts the drawn `x` — likely) or `"source": "gemini"` (fallback). Both are valid.

- [ ] **Step 6: Commit**

```
git add backend/main.py backend/tests/test_main.py
git commit -m "feat: gate /convert through local classifier; add source+confidence to response"
```

---

## Task 8: Frontend source badge

**Files:**
- Modify: `frontend/mathboard/src/App.jsx`
- Modify: `frontend/mathboard/src/App.css`

- [ ] **Step 1: Extend `convertResult` state in `App.jsx`**

In `handleConvert` (the function that calls `/convert`), find the `setConvertResult({ ... })` call and add two fields. The block currently looks roughly like:
```jsx
setConvertResult({
  latex: data.latex || "",
  solution: solutionText,
  isSolutionError: Boolean(solData.error),
  operationLabel: solData.operation_label || "",
  steps: solData.steps || [],
  latexResult: solData.latex_result || "",
});
```
Change to:
```jsx
setConvertResult({
  latex: data.latex || "",
  solution: solutionText,
  isSolutionError: Boolean(solData.error),
  operationLabel: solData.operation_label || "",
  steps: solData.steps || [],
  latexResult: solData.latex_result || "",
  source: data.source || "gemini",
  confidence: typeof data.confidence === "number" ? data.confidence : null,
});
```

- [ ] **Step 2: Render the source badge** in `App.jsx`

Find the JSX that renders the operation badge — it currently looks like:
```jsx
{convertResult.operationLabel && (
  <div className="operation-badge">{convertResult.operationLabel}</div>
)}
```
Replace with a wrapping flex row that includes both badges:
```jsx
<div className="badge-row">
  {convertResult.operationLabel && (
    <div className="operation-badge">{convertResult.operationLabel}</div>
  )}
  {convertResult.source === "local" ? (
    <div
      className="source-badge source-local"
      title={
        convertResult.confidence != null
          ? `Confidence: ${(convertResult.confidence * 100).toFixed(0)}%`
          : "Recognized locally"
      }
    >
      ⚡ Local model
    </div>
  ) : (
    <div className="source-badge source-gemini" title="Recognized by Gemini">
      ☁ Gemini
    </div>
  )}
</div>
```

- [ ] **Step 3: Add CSS** to `App.css`

Append to the file:
```css
/* ---------- Source badge ---------- */
.badge-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}

.source-badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 4px 10px;
  border-radius: 20px;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.4px;
  border: 1px solid var(--border);
  background: var(--surface);
}

.source-badge.source-local {
  color: var(--brand-1);
  border-color: var(--brand-1);
  background: rgba(102, 126, 234, 0.08);
}

.source-badge.source-gemini {
  color: var(--text-soft);
  background: var(--bg);
}

[data-theme="dark"] .source-badge.source-local {
  background: rgba(102, 126, 234, 0.15);
}
```

(Variable names — `--brand-1`, `--surface`, `--border`, `--bg`, `--text-soft` — must match what the existing theme actually defines. If the redesign used different names, substitute the equivalents from the top of `App.css`.)

- [ ] **Step 4: Verify lint + build**

Run from `frontend/mathboard/`:
```
npm run lint
npm run build
```
Expected: no errors, no new warnings.

- [ ] **Step 5: Manual smoke test**

Start backend (`uvicorn main:app --reload` from `backend/`) and frontend (`npm run dev` from `frontend/mathboard/`) if not already running. In a browser at http://localhost:5173, draw something simple (e.g., a single `x`) and click Convert. Confirm the result panel shows either `⚡ Local model` (if classifier accepts) or `☁ Gemini` badge.

- [ ] **Step 6: Commit**

```
git add frontend/mathboard/src/App.jsx frontend/mathboard/src/App.css
git commit -m "feat(ui): show source badge — local model vs Gemini — in result panel"
```

---

## Task 9: Documentation

**Files:**
- Create: `backend/ml/README.md`
- Modify: `SETUP.md`

- [ ] **Step 1: Write `backend/ml/README.md`**

Create the file:
```markdown
# Local Symbol Classifier

A small CNN that runs server-side and handles single-symbol drawings before
falling back to Gemini. Trained on the top-100 most-common symbol classes
in HASYv2.

## Train

From `backend/`:

```
.venv/Scripts/python -m ml.train
```

~30 minutes on CPU. Produces three files in `ml/artifacts/`:

| File | Purpose |
|---|---|
| `model.pt` | Trained `state_dict` for `SymbolCNN` (~1 MB) |
| `classes.json` | Index → LaTeX label mapping (100 entries) |
| `metrics.json` | Accuracy, coverage at confidence threshold, train-set mean/std |

`ml/artifacts/` is gitignored. Re-run training to refresh.

## How it gates

For each `/convert` request:

1. Decode image, count connected components.
2. If `> 4 components` → skip local model, go to Gemini.
3. Else: crop, resize 32×32, run model, take softmax max as confidence.
4. If `confidence ≥ 0.85` → accept locally, return without calling Gemini.
5. Else → fall through to Gemini.

If `model.pt` is missing or corrupt, the classifier reports
`is_loaded() == False` and every request goes to Gemini — the app keeps
working before training is done.

## Tuning

After training, look at `metrics.json`:

```json
{
  "val_top1_acc": 0.85,
  "coverage_at_threshold": 0.71,
  "accuracy_on_accepted": 0.95,
  ...
}
```

- `val_top1_acc` — how often the model's top-1 prediction is right on the held-out 15%.
- `coverage_at_threshold` — fraction of val samples accepted at the 0.85 confidence threshold.
- `accuracy_on_accepted` — accuracy on just the accepted slice. **This is the user-facing accuracy**; aim for ≥ 0.90.

If `accuracy_on_accepted < 0.90`, raise `CONFIDENCE_THRESHOLD` in
`backend/ml/classifier.py` (and also in `train.py` so the metrics
match) — fewer accepts, but the ones we do make are right more often.

## Files

| File | Purpose |
|---|---|
| `model.py` | `SymbolCNN` nn.Module |
| `preprocessing.py` | Otsu, components, crop, resize, normalize — used by both training and inference |
| `train.py` | CLI entry point, training loop, threshold sweep |
| `classifier.py` | `SymbolClassifier` — load artifacts, classify image bytes |
| `artifacts/` | Gitignored — produced by training |
```

- [ ] **Step 2: Add a section to `SETUP.md`**

Find the "## Running tests" section and insert the following just before it (or wherever it fits the existing flow):
```markdown
## Optional: train the local classifier

The backend has an optional local CNN that handles single-symbol drawings
without calling Gemini, falling back to Gemini for anything else. To enable it:

```
cd backend
.venv\Scripts\activate
python -m ml.train
```

~30 minutes on CPU. Produces `backend/ml/artifacts/{model.pt, classes.json, metrics.json}`.

The app keeps working without this step — every request just goes to Gemini.
See `backend/ml/README.md` for tuning details.
```

- [ ] **Step 3: Verify both files render**

Open in any markdown viewer or VS Code preview. Spot-check formatting.

- [ ] **Step 4: Commit**

```
git add backend/ml/README.md SETUP.md
git commit -m "docs: README for the local classifier; SETUP.md training section"
```

---

## Final verification

- [ ] **Run all backend tests**

```
cd backend && .venv/Scripts/python -m pytest tests/ -v
```
Expected: 45 passed (31 pre-existing + 11 classifier + 1 train smoke + 3 convert gating).

- [ ] **Run frontend lint + build**

```
cd frontend/mathboard && npm run lint && npm run build
```
Expected: no errors, no new warnings.

- [ ] **End-to-end check**

Start backend and frontend. Draw a `x` (or any digit/letter), click Convert.
Confirm the response includes a badge — either `⚡ Local model` or `☁ Gemini` —
and that the LaTeX + solution still display correctly.
