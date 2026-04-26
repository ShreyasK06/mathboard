# MNIST Hybrid + R Reintegration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the local classifier to recognize MNIST digits, add R Plumber as a parallel symbolic cross-solver (Ryacas), log every `/convert` call to SQLite, and replace the static HASYv2 viewer with a 4-tab Shiny dashboard showing real, current model + activity data.

**Architecture:** Two new R processes run alongside FastAPI — Plumber on :8003 (cross-solver, called per request from Python) and Shiny on :3838 (dashboard, reads activity.db + training artifacts). Both are optional; the app degrades gracefully if either is missing. Training pipeline grows a CombinedDataset (HASYv2 + MNIST) and emits a confusion matrix the dashboard renders.

**Tech Stack:** PyTorch (CPU), Pillow, numpy, FastAPI (existing), SQLite (stdlib), R + Plumber + Ryacas + Shiny + ggplot2 + RSQLite, requests.

**Spec:** [`docs/superpowers/specs/2026-04-25-mnist-and-r-reintegration-design.md`](../specs/2026-04-25-mnist-and-r-reintegration-design.md)

---

## File Structure

```
backend/
  plumber.R                    — NEW: R Plumber API (cross-solver), port 8003
  run_plumber.r                — NEW: launcher, `Rscript run_plumber.r`
  shiny_app.R                  — REWRITTEN: 4-tab dashboard (HASYv2 viewer removed)
  ryacas_client.py             — NEW: Python client for Plumber, never raises
  activity_log.py              — NEW: SQLite logger for /convert, never raises
  activity.db                  — NEW (gitignored): created on first log_request
  ml/
    idx_parser.py              — NEW: minimal IDX file reader for MNIST
    datasets.py                — NEW: HasyDataset, MnistDataset, CombinedDataset
    confusion.py               — NEW: build + save confusion matrix
    train.py                   — MODIFIED: uses CombinedDataset, saves confusion.json
    artifacts/
      confusion.json           — NEW (gitignored): produced by training
  data/mnist/                  — NEW (gitignored): auto-downloaded IDX files
  main.py                      — MODIFIED: parallel SymPy+Ryacas, activity logging
  tests/
    test_idx_parser.py         — NEW
    test_datasets.py           — NEW
    test_ryacas_client.py      — NEW
    test_activity_log.py       — NEW
    test_plumber.R             — NEW
    test_main.py               — MODIFIED: extended for new response fields

frontend/mathboard/src/
  App.jsx                      — MODIFIED: agreement line, tab label
  App.css                      — MODIFIED: agreement styles

SETUP.md                       — MODIFIED: 3rd terminal now runs plumber + shiny
```

Order of implementation matches dependencies: data → training → activity log → Ryacas (R + Python) → main.py wire-up → frontend → dashboard → docs.

---

## Task 1: IDX parser for MNIST

**Files:**
- Create: `backend/ml/idx_parser.py`
- Create: `backend/tests/test_idx_parser.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_idx_parser.py`:
```python
import gzip
import struct
from pathlib import Path

import numpy as np

from ml.idx_parser import read_idx_images, read_idx_labels


def _write_idx_images(path: Path, arr: np.ndarray) -> None:
    n, h, w = arr.shape
    header = struct.pack(">IIII", 0x00000803, n, h, w)
    with gzip.open(path, "wb") as f:
        f.write(header + arr.astype(np.uint8).tobytes())


def _write_idx_labels(path: Path, labels: np.ndarray) -> None:
    n = labels.shape[0]
    header = struct.pack(">II", 0x00000801, n)
    with gzip.open(path, "wb") as f:
        f.write(header + labels.astype(np.uint8).tobytes())


def test_read_idx_images_returns_correct_shape(tmp_path):
    arr = (np.random.rand(7, 28, 28) * 255).astype(np.uint8)
    p = tmp_path / "imgs.idx.gz"
    _write_idx_images(p, arr)
    out = read_idx_images(p)
    assert out.shape == (7, 28, 28)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, arr)


def test_read_idx_labels_returns_correct_shape(tmp_path):
    labels = np.array([0, 5, 9, 3, 1], dtype=np.uint8)
    p = tmp_path / "labels.idx.gz"
    _write_idx_labels(p, labels)
    out = read_idx_labels(p)
    assert out.shape == (5,)
    np.testing.assert_array_equal(out, labels)


def test_read_idx_images_handles_uncompressed(tmp_path):
    arr = (np.random.rand(3, 28, 28) * 255).astype(np.uint8)
    p = tmp_path / "imgs.idx"
    header = struct.pack(">IIII", 0x00000803, 3, 28, 28)
    p.write_bytes(header + arr.tobytes())
    out = read_idx_images(p)
    assert out.shape == (3, 28, 28)
```

- [ ] **Step 2: Run tests, expect failure**

Run from `backend/`:
```
.venv/Scripts/python -m pytest tests/test_idx_parser.py -v
```
Expected: `ModuleNotFoundError: No module named 'ml.idx_parser'`

- [ ] **Step 3: Implement `backend/ml/idx_parser.py`**

```python
"""Minimal MNIST IDX file parser.

The IDX format header is:
- magic number (4 bytes, big-endian): 0x00000803 for images, 0x00000801 for labels
- count (4 bytes, big-endian)
- for images only: rows (4 bytes), cols (4 bytes)
- pixel/label bytes follow

Files may be gzipped (.gz) or raw.
"""

import gzip
import struct
from pathlib import Path

import numpy as np


def _open(path: Path):
    return gzip.open(path, "rb") if str(path).endswith(".gz") else open(path, "rb")


def read_idx_images(path: Path) -> np.ndarray:
    with _open(path) as f:
        magic, n, h, w = struct.unpack(">IIII", f.read(16))
        if magic != 0x00000803:
            raise ValueError(f"Not an IDX images file (magic={magic:#010x})")
        data = np.frombuffer(f.read(n * h * w), dtype=np.uint8)
        return data.reshape(n, h, w)


def read_idx_labels(path: Path) -> np.ndarray:
    with _open(path) as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 0x00000801:
            raise ValueError(f"Not an IDX labels file (magic={magic:#010x})")
        return np.frombuffer(f.read(n), dtype=np.uint8)
```

- [ ] **Step 4: Run tests, expect pass**

```
.venv/Scripts/python -m pytest tests/test_idx_parser.py -v
```
Expected: 3 passed

- [ ] **Step 5: Commit**

```
git add backend/ml/idx_parser.py backend/tests/test_idx_parser.py
git commit -m "feat(ml): add minimal MNIST IDX file parser (no torchvision dep)"
```

---

## Task 2: Datasets module (HASYv2 + MNIST + Combined)

**Files:**
- Create: `backend/ml/datasets.py`
- Create: `backend/tests/test_datasets.py`
- Modify: `backend/ml/train.py` (only deletes `HasyDataset` — re-imports it from `datasets.py`)

The current `HasyDataset` lives inside `train.py`. We're moving it into `ml/datasets.py` so MNIST, HASYv2, and the combined dataset all live together. `train.py` will import them in Task 3.

- [ ] **Step 1: Write failing tests** at `backend/tests/test_datasets.py`

```python
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ml.datasets import (
    CombinedDataset,
    HasyDataset,
    MnistDataset,
    build_weighted_sampler,
)


# ---- helpers ---------------------------------------------------------------

def _make_synthetic_hasy(root: Path, classes, samples_per_class=5):
    images_dir = root / "hasy-data"
    images_dir.mkdir(parents=True, exist_ok=True)
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
    with open(root / "hasy-data-labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "symbol_id", "latex", "user_id"])
        w.writeheader()
        w.writerows(rows)
    return rows


def _make_synthetic_mnist_arrays():
    images = np.zeros((20, 28, 28), dtype=np.uint8)
    images[:, 5:23, 5:23] = 200  # bright "ink" in middle (MNIST polarity)
    labels = np.tile(np.arange(10), 2).astype(np.uint8)
    return images, labels


# ---- HasyDataset ----------------------------------------------------------

def test_hasy_dataset_loads_correct_shape(tmp_path):
    classes = ["A", "B"]
    _make_synthetic_hasy(tmp_path, classes, samples_per_class=3)
    rows = [(f"hasy-data/v2-{i:05d}.png", i % 2) for i in range(6)]
    ds = HasyDataset(rows, image_root=tmp_path, augment=False, mean=0.5, std=0.5)
    assert len(ds) == 6
    x, y = ds[0]
    assert x.shape == (1, 32, 32)
    assert isinstance(y, int)


# ---- MnistDataset ---------------------------------------------------------

def test_mnist_dataset_polarity_inverted_to_match_hasyv2(tmp_path):
    images, labels = _make_synthetic_mnist_arrays()
    ds = MnistDataset(images, labels, augment=False, mean=0.5, std=0.5, label_offset=100)
    x, y = ds[0]
    # MNIST input had bright pixels in middle (200). After invert + normalize,
    # those positions should become NEGATIVE values (dark ink).
    middle = float(x[0, 12:20, 12:20].mean())
    edge = float(x[0, 0:4, 0:4].mean())
    assert middle < edge, f"middle should be darker than edge after inversion (got middle={middle}, edge={edge})"


def test_mnist_dataset_resized_to_32x32(tmp_path):
    images, labels = _make_synthetic_mnist_arrays()
    ds = MnistDataset(images, labels, augment=False, mean=0.5, std=0.5, label_offset=0)
    x, _ = ds[0]
    assert x.shape == (1, 32, 32)


def test_mnist_dataset_applies_label_offset():
    images, labels = _make_synthetic_mnist_arrays()
    ds = MnistDataset(images, labels, augment=False, mean=0.5, std=0.5, label_offset=100)
    seen = {ds[i][1] for i in range(len(ds))}
    assert seen == set(range(100, 110))


# ---- CombinedDataset ------------------------------------------------------

def test_combined_dataset_disjoint_label_indices(tmp_path):
    classes = ["A", "B"]
    _make_synthetic_hasy(tmp_path, classes, samples_per_class=3)
    rows = [(f"hasy-data/v2-{i:05d}.png", i % 2) for i in range(6)]
    hasy = HasyDataset(rows, image_root=tmp_path, augment=False, mean=0.5, std=0.5)

    images, labels = _make_synthetic_mnist_arrays()
    mnist = MnistDataset(images, labels, augment=False, mean=0.5, std=0.5, label_offset=2)

    combined = CombinedDataset([hasy, mnist])
    assert len(combined) == len(hasy) + len(mnist)
    seen = {combined[i][1] for i in range(len(combined))}
    assert seen.issubset(set(range(2 + 10)))
    # both source ranges are present
    assert any(s < 2 for s in seen)   # HASYv2 labels 0,1
    assert any(s >= 2 for s in seen)  # MNIST labels 2..11


# ---- WeightedRandomSampler -----------------------------------------------

def test_weighted_sampler_balances_classes(tmp_path):
    """Class with 1000 samples and class with 100 samples should both be sampled
    roughly equally over many draws."""
    labels = [0] * 1000 + [1] * 100
    sampler = build_weighted_sampler(labels, num_samples=4400, seed=42)
    drawn = [labels[i] for i in sampler]
    c0 = sum(1 for l in drawn if l == 0)
    c1 = sum(1 for l in drawn if l == 1)
    # Expect roughly 50/50; allow 20% slop
    assert 0.4 * len(drawn) < c0 < 0.6 * len(drawn)
    assert 0.4 * len(drawn) < c1 < 0.6 * len(drawn)
```

- [ ] **Step 2: Run tests, expect failure**

```
.venv/Scripts/python -m pytest tests/test_datasets.py -v
```
Expected: ImportError for `ml.datasets`.

- [ ] **Step 3: Implement `backend/ml/datasets.py`**

```python
"""Dataset classes for HASYv2, MNIST, and the combined training set.

Both single-source datasets emit (1, 32, 32) float tensors with HASYv2's
polarity (ink dark on bright background). HASYv2 is loaded as-is; MNIST
is inverted (255 - pixel) and resized 28x28 -> 32x32 to match.
"""

import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler


class HasyDataset(Dataset):
    """Loads HASYv2 PNGs from disk. Each row is (relative_path, label_idx)."""

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
            img2 = Image.fromarray((arr * 255).astype(np.uint8), mode="L").resize(
                (32, 32), Image.LANCZOS
            )
            arr = np.asarray(img2, dtype=np.float32) / 255.0
        return arr

    def _augment(self, arr: np.ndarray) -> np.ndarray:
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
        return t, int(label)


class MnistDataset(Dataset):
    """In-memory MNIST. Inverts polarity and resizes to 32x32 to match HASYv2."""

    def __init__(
        self,
        images: np.ndarray,           # (N, 28, 28) uint8, white-ink-on-black
        labels: np.ndarray,           # (N,) uint8
        augment: bool,
        mean: float,
        std: float,
        label_offset: int,            # added to every label so MNIST labels live in [offset, offset+10)
    ) -> None:
        if images.ndim != 3:
            raise ValueError(f"images must be (N, H, W); got {images.shape}")
        self.images = images
        self.labels = labels
        self.augment = augment
        self.mean = mean
        self.std = std
        self.label_offset = label_offset

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def _to_32x32_dark_on_bright(self, raw28: np.ndarray) -> np.ndarray:
        # Invert (white-ink-on-black -> black-ink-on-white) then resize to 32x32.
        inverted = 255 - raw28
        img = Image.fromarray(inverted, mode="L").resize((32, 32), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0

    def _augment(self, arr: np.ndarray) -> np.ndarray:
        img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
        angle = random.uniform(-10, 10)
        tx, ty = random.randint(-2, 2), random.randint(-2, 2)
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=255)
        img = img.transform(
            img.size, Image.AFFINE, (1, 0, tx, 0, 1, ty), fillcolor=255
        )
        return np.asarray(img, dtype=np.float32) / 255.0

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        arr = self._to_32x32_dark_on_bright(self.images[i])
        if self.augment:
            arr = self._augment(arr)
        t = torch.from_numpy(arr).unsqueeze(0)
        t = (t - self.mean) / self.std
        return t, int(self.labels[i]) + self.label_offset


class CombinedDataset(Dataset):
    """Concatenates multiple datasets while preserving the original index ranges."""

    def __init__(self, sources: Sequence[Dataset]) -> None:
        self.sources = list(sources)
        self.lengths = [len(s) for s in self.sources]
        self.cumulative = np.cumsum([0] + self.lengths)
        self._total = int(self.cumulative[-1])

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, i: int):
        if i < 0 or i >= self._total:
            raise IndexError(i)
        # find which source owns index i
        src_idx = int(np.searchsorted(self.cumulative, i, side="right") - 1)
        local_i = i - int(self.cumulative[src_idx])
        return self.sources[src_idx][local_i]


def build_weighted_sampler(
    labels: Iterable[int], num_samples: int, seed: int
) -> WeightedRandomSampler:
    """One sample weight = 1 / class_count. Each batch sees roughly balanced classes."""
    label_arr = np.asarray(list(labels))
    counts = np.bincount(label_arr)
    weights = 1.0 / np.where(counts == 0, 1, counts)[label_arr]
    g = torch.Generator()
    g.manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).float(),
        num_samples=num_samples,
        replacement=True,
        generator=g,
    )
```

- [ ] **Step 4: Run tests, expect pass**

```
.venv/Scripts/python -m pytest tests/test_datasets.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Run full test suite to confirm nothing else broke**

```
.venv/Scripts/python -m pytest tests/ 2>&1 | tail -5
```
Expected: existing 45 tests still pass (Task 3 will switch train.py to import from datasets.py; for now both definitions of HasyDataset coexist, but train.py's local class is what test_train_smoke.py uses, so it should still pass).

- [ ] **Step 6: Commit**

```
git add backend/ml/datasets.py backend/tests/test_datasets.py
git commit -m "feat(ml): add HasyDataset/MnistDataset/CombinedDataset + weighted sampler"
```

---

## Task 3: Switch train.py to use datasets.py + MNIST + emit confusion matrix

**Files:**
- Create: `backend/ml/confusion.py`
- Modify: `backend/ml/train.py`
- Modify: `backend/tests/test_train_smoke.py` (verify confusion.json + still works without MNIST flag)

- [ ] **Step 1: Implement `backend/ml/confusion.py`**

```python
"""Build and save a confusion matrix from a trained model + val loader."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_confusion(
    model: torch.nn.Module,
    val_loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> list[list[int]]:
    model.eval()
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        for true_label, pred_label in zip(y.tolist(), pred.tolist()):
            matrix[true_label, pred_label] += 1
    return matrix.tolist()


def save_confusion(path: Path, classes: list[str], matrix: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "matrix": matrix}, f)
```

- [ ] **Step 2: Add an MNIST downloader function** at `backend/ml/datasets.py`

Append to the bottom of the file:
```python
import urllib.request

MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}

MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download_mnist(dest_dir: Path) -> dict[str, Path]:
    """Download MNIST IDX files if not already present. Returns the file paths."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, fname in MNIST_FILES.items():
        p = dest_dir / fname
        if not p.exists():
            print(f"[mnist] downloading {fname} ...", flush=True)
            urllib.request.urlretrieve(MNIST_URLS[key], p)
        paths[key] = p
    return paths


def load_mnist(dest_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Download (if needed) and concatenate train+test into one (N,28,28) + (N,) array."""
    from ml.idx_parser import read_idx_images, read_idx_labels
    paths = download_mnist(dest_dir)
    train_imgs = read_idx_images(paths["train_images"])
    train_lbls = read_idx_labels(paths["train_labels"])
    test_imgs = read_idx_images(paths["test_images"])
    test_lbls = read_idx_labels(paths["test_labels"])
    return (
        np.concatenate([train_imgs, test_imgs], axis=0),
        np.concatenate([train_lbls, test_lbls], axis=0),
    )
```

- [ ] **Step 3: Rewrite `backend/ml/train.py`** to use the new modules + MNIST

Replace the file contents with:
```python
"""Train a SymbolCNN on HASYv2 top-N classes plus optionally MNIST digits.

Run from backend/:
    python -m ml.train                  # default: includes MNIST
    python -m ml.train --no-mnist       # HASYv2 only
"""

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from ml.confusion import compute_confusion, save_confusion
from ml.datasets import (
    CombinedDataset,
    HasyDataset,
    MnistDataset,
    build_weighted_sampler,
    load_mnist,
)
from ml.model import SymbolCNN

CONFIDENCE_THRESHOLD = 0.85
DEFAULT_TOP_N = 100
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_SEED = 42
DEFAULT_VAL_FRAC = 0.15


def _read_labels(labels_csv: Path) -> list[dict]:
    with open(labels_csv, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pick_top_classes(rows: list[dict], top_n: int) -> list[str]:
    counts = Counter(r["latex"] for r in rows)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in ordered[:top_n]]


def _stratified_split(rows, val_frac, seed):
    rng = random.Random(seed)
    by_label: dict[int, list] = {}
    for r in rows:
        by_label.setdefault(r[1], []).append(r)
    train, val = [], []
    for _label, items in by_label.items():
        rng.shuffle(items)
        cut = max(1, int(len(items) * val_frac))
        val.extend(items[:cut])
        train.extend(items[cut:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _stats_from_arrays(arrays: list[np.ndarray]) -> tuple[float, float]:
    """Mean and std across a sample of pre-normalized [0,1] images."""
    flat = np.concatenate([a.flatten() for a in arrays])
    return float(flat.mean()), float(flat.std() or 1e-6)


def _train_one_epoch(model, loader, optim, criterion, device):
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
def _evaluate(model, loader, criterion, device):
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
    include_mnist: bool = True,
    mnist_dir: Path | None = None,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # ---- HASYv2 ------------------------------------------------------------
    raw = _read_labels(data_root / "hasy-data-labels.csv")
    print(f"[train] HASYv2: {len(raw):,} rows", flush=True)
    hasy_classes = _pick_top_classes(raw, top_n_classes)
    cls_to_idx = {c: i for i, c in enumerate(hasy_classes)}
    hasy_rows = [(r["path"], cls_to_idx[r["latex"]]) for r in raw if r["latex"] in cls_to_idx]
    print(f"[train] HASYv2: top {len(hasy_classes)} classes, {len(hasy_rows):,} samples", flush=True)
    h_train, h_val = _stratified_split(hasy_rows, DEFAULT_VAL_FRAC, seed)

    # ---- MNIST -------------------------------------------------------------
    classes = list(hasy_classes)
    mnist_train, mnist_val = None, None
    label_offset = len(hasy_classes)
    if include_mnist:
        mnist_dir = mnist_dir or (data_root.parent / "mnist")
        print("[train] loading MNIST...", flush=True)
        m_imgs, m_lbls = load_mnist(mnist_dir)
        # stratified split by label
        train_idx, val_idx = [], []
        rng = random.Random(seed)
        for d in range(10):
            idxs = list(np.where(m_lbls == d)[0])
            rng.shuffle(idxs)
            cut = max(1, int(len(idxs) * DEFAULT_VAL_FRAC))
            val_idx.extend(idxs[:cut])
            train_idx.extend(idxs[cut:])
        m_train_imgs = m_imgs[train_idx]; m_train_lbls = m_lbls[train_idx]
        m_val_imgs = m_imgs[val_idx]; m_val_lbls = m_lbls[val_idx]
        classes = list(hasy_classes) + [str(d) for d in range(10)]
        print(f"[train] MNIST: {len(train_idx):,} train / {len(val_idx):,} val", flush=True)

    # ---- compute combined train-set stats ---------------------------------
    print("[train] sampling images for mean/std...", flush=True)
    sample_arrays: list[np.ndarray] = []
    for rel, _ in h_train[:1500]:
        with Image.open(data_root / rel) as img:
            sample_arrays.append(np.asarray(img.convert("L"), dtype=np.float32) / 255.0)
    if include_mnist:
        for raw28 in m_train_imgs[:500]:
            inv = 255 - raw28
            big = np.asarray(Image.fromarray(inv, mode="L").resize((32, 32), Image.BILINEAR), dtype=np.float32) / 255.0
            sample_arrays.append(big)
    mean, std = _stats_from_arrays(sample_arrays)
    print(f"[train] dataset mean={mean:.4f} std={std:.4f}", flush=True)

    # ---- build datasets ----------------------------------------------------
    hasy_train_ds = HasyDataset(h_train, image_root=data_root, augment=True, mean=mean, std=std)
    hasy_val_ds = HasyDataset(h_val, image_root=data_root, augment=False, mean=mean, std=std)
    train_sources = [hasy_train_ds]
    val_sources = [hasy_val_ds]
    train_label_seq = [r[1] for r in h_train]
    if include_mnist:
        m_train_ds = MnistDataset(m_train_imgs, m_train_lbls, augment=True, mean=mean, std=std, label_offset=label_offset)
        m_val_ds = MnistDataset(m_val_imgs, m_val_lbls, augment=False, mean=mean, std=std, label_offset=label_offset)
        train_sources.append(m_train_ds)
        val_sources.append(m_val_ds)
        train_label_seq = train_label_seq + [int(l) + label_offset for l in m_train_lbls]

    train_combined = CombinedDataset(train_sources)
    val_combined = CombinedDataset(val_sources)

    sampler = build_weighted_sampler(train_label_seq, num_samples=len(train_combined), seed=seed)
    train_loader = DataLoader(train_combined, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_combined, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SymbolCNN(num_classes=len(classes)).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, epochs))
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_state, best_metrics = None, {}
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _train_one_epoch(model, train_loader, optim, criterion, device)
        val_loss, val_acc, coverage, acc_accepted = _evaluate(model, val_loader, criterion, device)
        scheduler.step()
        print(
            f"[train] epoch {epoch}/{epochs} train_acc={tr_acc:.4f} val_acc={val_acc:.4f} "
            f"coverage@{CONFIDENCE_THRESHOLD:.2f}={coverage:.4f} acc_on_accepted={acc_accepted:.4f}",
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
                "includes_mnist": include_mnist,
            }

    torch.save(best_state, artifacts_dir / "model.pt")
    with open(artifacts_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
    with open(artifacts_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)

    # confusion matrix on best model
    print("[train] computing confusion matrix...", flush=True)
    model.load_state_dict(best_state)
    matrix = compute_confusion(model, val_loader, num_classes=len(classes), device=device)
    save_confusion(artifacts_dir / "confusion.json", classes=classes, matrix=matrix)

    print(f"[train] Best val_acc={best_acc:.4f}. Wrote artifacts to {artifacts_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/hasy", type=Path)
    parser.add_argument("--artifacts-dir", default="ml/artifacts", type=Path)
    parser.add_argument("--top-n", default=DEFAULT_TOP_N, type=int)
    parser.add_argument("--epochs", default=DEFAULT_EPOCHS, type=int)
    parser.add_argument("--batch-size", default=DEFAULT_BATCH_SIZE, type=int)
    parser.add_argument("--seed", default=DEFAULT_SEED, type=int)
    parser.add_argument("--mnist-dir", default=None, type=Path)
    parser.add_argument("--no-mnist", action="store_true")
    args = parser.parse_args()
    run(
        data_root=args.data_root,
        artifacts_dir=args.artifacts_dir,
        top_n_classes=args.top_n,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        include_mnist=not args.no_mnist,
        mnist_dir=args.mnist_dir,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Update the smoke test** at `backend/tests/test_train_smoke.py`

Replace its contents with:
```python
"""Smoke test for ml/train.py — runs 1 epoch on a synthetic dataset, no MNIST."""

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

import ml.train as train_module


def _make_synthetic_dataset(root, classes, samples_per_class):
    images_dir = root / "hasy-data"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(0)
    idx = 0
    for cls in classes:
        for _ in range(samples_per_class):
            arr = (rng.random((32, 32)) * 255).astype(np.uint8)
            fname = f"v2-{idx:05d}.png"
            Image.fromarray(arr, mode="L").save(images_dir / fname)
            rows.append({"path": f"hasy-data/{fname}", "symbol_id": str(idx),
                         "latex": cls, "user_id": "0"})
            idx += 1
    with open(root / "hasy-data-labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "symbol_id", "latex", "user_id"])
        w.writeheader(); w.writerows(rows)
    return root


def test_training_smoke_hasy_only(tmp_path):
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
        include_mnist=False,
    )

    for fname in ("model.pt", "classes.json", "metrics.json", "confusion.json"):
        assert (artifacts_dir / fname).exists(), f"missing {fname}"

    metrics = json.loads((artifacts_dir / "metrics.json").read_text(encoding="utf-8"))
    for key in ("val_top1_acc", "coverage_at_threshold", "accuracy_on_accepted",
                "train_mean", "train_std"):
        assert key in metrics

    confusion = json.loads((artifacts_dir / "confusion.json").read_text(encoding="utf-8"))
    assert "classes" in confusion and "matrix" in confusion
    assert len(confusion["classes"]) == 5
    assert len(confusion["matrix"]) == 5 and len(confusion["matrix"][0]) == 5
```

- [ ] **Step 5: Run smoke test, expect pass**

```
.venv/Scripts/python -m pytest tests/test_train_smoke.py -v
```
Expected: 1 passed in ~10-20 seconds.

- [ ] **Step 6: Run full test suite**

```
.venv/Scripts/python -m pytest tests/ 2>&1 | tail -5
```
Expected: all tests pass (was 45, now 45 + 3 idx_parser + 6 datasets = 54).

- [ ] **Step 7: Commit**

```
git add backend/ml/train.py backend/ml/confusion.py backend/ml/datasets.py backend/tests/test_train_smoke.py
git commit -m "feat(ml): train.py uses CombinedDataset (HASYv2 + MNIST), saves confusion.json"
```

---

## Task 4: Run real training (HASYv2 + MNIST)

This task is a manual run, not TDD. Produces the artifacts the classifier loads at runtime.

- [ ] **Step 1: Verify HASYv2 data is present**

From `backend/`:
```
ls data/hasy/hasy-data | head -3 && ls data/hasy/hasy-data-labels.csv
```
Expected: at least 3 PNGs + CSV. (HASYv2 was downloaded earlier in the project.)

- [ ] **Step 2: Train (downloads MNIST on first run)**

From `backend/`:
```
.venv/Scripts/python -m ml.train
```
Expected: ~30-45 minutes on CPU (slightly longer than HASYv2-only because MNIST adds ~70k samples). MNIST downloads to `data/mnist/` (~12 MB) on first run. Per-epoch lines, then:
```
[train] computing confusion matrix...
[train] Best val_acc=0.XX. Wrote artifacts to ml/artifacts
```

- [ ] **Step 3: Sanity-check artifacts**

From `backend/`:
```
.venv/Scripts/python -c "import json; m = json.load(open('ml/artifacts/metrics.json')); print(json.dumps(m, indent=2))"
.venv/Scripts/python -c "import json; c = json.load(open('ml/artifacts/classes.json')); print(len(c), 'classes; first 5:', c[:5], '; last 11:', c[-11:])"
ls -lh ml/artifacts/model.pt ml/artifacts/confusion.json
```
Expected:
- `metrics.json` has `includes_mnist: true`, `num_classes: 110`, `val_top1_acc >= 0.85`, `accuracy_on_accepted >= 0.92`.
- `classes.json` has 110 entries, last 10 are `"0"..."9"`.
- `model.pt` ~2-3 MB; `confusion.json` ~few KB to 100 KB depending on class counts.

- [ ] **Step 4: No commit (artifacts gitignored)**

```
git status
```
Expected: clean working tree.

---

## Task 5: Activity logging (SQLite)

**Files:**
- Create: `backend/activity_log.py`
- Create: `backend/tests/test_activity_log.py`
- Modify: `backend/.gitignore` (add `activity.db`)

- [ ] **Step 1: Add `activity.db` to `backend/.gitignore`**

Append `activity.db` to the file. Resulting file content:
```
ml/artifacts/
__pycache__/
*.pyc
.venv/
.env
.pytest_cache/
activity.db
```

- [ ] **Step 2: Write failing tests** at `backend/tests/test_activity_log.py`

```python
import io
import sqlite3
from pathlib import Path

from PIL import Image

import activity_log


def _png_bytes(size=(40, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, format="PNG")
    return buf.getvalue()


def _open(db: Path) -> sqlite3.Connection:
    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    return c


def test_creates_schema_on_first_use(tmp_path):
    db = tmp_path / "a.db"
    activity_log.log_request(
        db_path=db,
        source="local", recognized_latex="x", confidence=0.9, num_components=1,
        operation="simplify", sympy_solution="x", ryacas_solution="x",
        agreement="match", duration_ms=42, image_bytes=_png_bytes(),
    )
    assert db.exists()
    with _open(db) as c:
        row = c.execute("SELECT * FROM requests").fetchone()
    assert row["source"] == "local"
    assert row["confidence"] == 0.9
    assert row["agreement"] == "match"


def test_log_then_read_roundtrip(tmp_path):
    db = tmp_path / "a.db"
    activity_log.log_request(
        db_path=db,
        source="gemini", recognized_latex="x+5=12", confidence=None, num_components=4,
        operation="solve", sympy_solution="7", ryacas_solution="7",
        agreement="match", duration_ms=812, image_bytes=_png_bytes(),
    )
    with _open(db) as c:
        row = c.execute("SELECT * FROM requests").fetchone()
    assert row["recognized_latex"] == "x+5=12"
    assert row["confidence"] is None
    assert row["operation"] == "solve"
    assert row["thumbnail_b64"] is not None
    assert row["thumbnail_b64"].startswith("iVBOR") or row["thumbnail_b64"].startswith("data:")


def test_thumbnail_is_64x64(tmp_path):
    import base64
    db = tmp_path / "a.db"
    activity_log.log_request(
        db_path=db,
        source="local", recognized_latex="x", confidence=0.9, num_components=1,
        operation="simplify", sympy_solution=None, ryacas_solution=None,
        agreement="match", duration_ms=42, image_bytes=_png_bytes(size=(200, 200)),
    )
    with _open(db) as c:
        b64 = c.execute("SELECT thumbnail_b64 FROM requests").fetchone()[0]
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw))
    assert img.size == (64, 64)


def test_cap_enforced(tmp_path):
    db = tmp_path / "a.db"
    for i in range(15):
        activity_log.log_request(
            db_path=db, source="local", recognized_latex=str(i), confidence=0.9,
            num_components=1, operation="simplify", sympy_solution=str(i),
            ryacas_solution=str(i), agreement="match", duration_ms=10,
            image_bytes=_png_bytes(), max_rows=10,
        )
    with _open(db) as c:
        n = c.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        last = c.execute("SELECT recognized_latex FROM requests ORDER BY id DESC LIMIT 1").fetchone()[0]
    assert n == 10
    assert last == "14"


def test_recovers_from_corrupt_db(tmp_path):
    db = tmp_path / "a.db"
    db.write_bytes(b"not a sqlite database at all")
    activity_log.log_request(
        db_path=db, source="local", recognized_latex="x", confidence=0.9,
        num_components=1, operation="simplify", sympy_solution="x",
        ryacas_solution="x", agreement="match", duration_ms=10, image_bytes=_png_bytes(),
    )
    # should now be a valid db with one row
    with _open(db) as c:
        n = c.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
    assert n == 1


def test_never_raises_on_failure(tmp_path):
    """If we can't write, log_request swallows the error."""
    activity_log.log_request(
        db_path=tmp_path / "definitely" / "missing" / "subdir" / "a.db",
        source="local", recognized_latex="x", confidence=0.9, num_components=1,
        operation="simplify", sympy_solution="x", ryacas_solution="x",
        agreement="match", duration_ms=10, image_bytes=_png_bytes(),
    )
    # no assertion; passing means no exception escaped
```

- [ ] **Step 3: Run tests, expect failure**

```
.venv/Scripts/python -m pytest tests/test_activity_log.py -v
```
Expected: ImportError for `activity_log`.

- [ ] **Step 4: Implement `backend/activity_log.py`**

```python
"""SQLite logger for /convert calls. Never raises — logging failures must
not break the user's request.
"""

import base64
import io
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_MAX_ROWS = 5000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp       TEXT    NOT NULL,
  source          TEXT    NOT NULL,
  recognized_latex TEXT   NOT NULL,
  confidence      REAL,
  num_components  INTEGER,
  operation       TEXT,
  sympy_solution  TEXT,
  ryacas_solution TEXT,
  agreement       TEXT,
  duration_ms     INTEGER,
  thumbnail_b64   TEXT
);
CREATE INDEX IF NOT EXISTS idx_timestamp ON requests(timestamp);
CREATE INDEX IF NOT EXISTS idx_source ON requests(source);
"""


def _make_thumbnail_b64(image_bytes: bytes, size: int = 64) -> Optional[str]:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.thumbnail((size, size), Image.LANCZOS)
        # if the input wasn't square, paste centered onto a (size x size) canvas
        canvas = Image.new("RGB", (size, size), "white")
        ox = (size - img.width) // 2
        oy = (size - img.height) // 2
        canvas.paste(img, (ox, oy))
        buf = io.BytesIO()
        canvas.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as exc:
        logger.warning("thumbnail failed: %s", exc)
        return None


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.executescript(_SCHEMA)
    return conn


def _try_recover_corrupt(db_path: Path) -> sqlite3.Connection:
    """Delete the file and recreate an empty DB."""
    try:
        db_path.unlink()
    except FileNotFoundError:
        pass
    return _connect(db_path)


def log_request(
    db_path: Path,
    *,
    source: str,
    recognized_latex: str,
    confidence: Optional[float],
    num_components: Optional[int],
    operation: Optional[str],
    sympy_solution: Optional[str],
    ryacas_solution: Optional[str],
    agreement: Optional[str],
    duration_ms: int,
    image_bytes: bytes,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> None:
    """Append one row to activity.db. Swallows all exceptions."""
    try:
        try:
            conn = _connect(db_path)
        except sqlite3.DatabaseError:
            conn = _try_recover_corrupt(db_path)
        with conn:
            conn.execute(
                """INSERT INTO requests (
                    timestamp, source, recognized_latex, confidence, num_components,
                    operation, sympy_solution, ryacas_solution, agreement, duration_ms,
                    thumbnail_b64
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    source,
                    recognized_latex,
                    confidence,
                    num_components,
                    operation,
                    sympy_solution,
                    ryacas_solution,
                    agreement,
                    duration_ms,
                    _make_thumbnail_b64(image_bytes),
                ),
            )
            # cap to last max_rows
            conn.execute(
                """DELETE FROM requests WHERE id NOT IN
                   (SELECT id FROM requests ORDER BY id DESC LIMIT ?)""",
                (max_rows,),
            )
        conn.close()
    except Exception as exc:
        print(f"[activity_log] failed: {exc}", file=sys.stderr, flush=True)
```

- [ ] **Step 5: Run tests, expect pass**

```
.venv/Scripts/python -m pytest tests/test_activity_log.py -v
```
Expected: 6 passed.

- [ ] **Step 6: Commit**

```
git add backend/.gitignore backend/activity_log.py backend/tests/test_activity_log.py
git commit -m "feat: SQLite activity logger for /convert calls (capped, self-recovering)"
```

---

## Task 6: R Plumber API (cross-solver)

**Files:**
- Create: `backend/plumber.R`
- Create: `backend/run_plumber.r`
- Create: `backend/tests/test_plumber.R`

- [ ] **Step 1: Write the failing test** at `backend/tests/test_plumber.R`

```r
library(testthat)
source("../plumber.R")

test_that("latex_to_yacas converts basic operators", {
  expect_equal(latex_to_yacas("x + 5"), "x + 5")
  expect_match(latex_to_yacas("\\frac{1}{2}"), "1/2", fixed = TRUE)
  expect_match(latex_to_yacas("x^2"), "x^2", fixed = TRUE)
})

test_that("detect_operation classifies expressions", {
  expect_equal(detect_operation("x + 5 = 12"), "solve")
  expect_equal(detect_operation("\\int x dx"), "integrate")
  expect_equal(detect_operation("\\frac{d}{dx} x^2"), "differentiate")
  expect_equal(detect_operation("\\lim_{x \\to 0} x"), "limit")
  expect_equal(detect_operation("x^2 + 2x + 1"), "simplify")
})

test_that("solve_with_ryacas returns a valid result for x + 5 = 12", {
  result <- solve_with_ryacas("x + 5 = 12")
  expect_equal(result$status, "success")
  expect_equal(result$operation, "solve")
  expect_match(result$solution, "7", fixed = TRUE)
})

test_that("solve_with_ryacas returns a valid result for simplification", {
  result <- solve_with_ryacas("x^2 + 2*x + 1")
  expect_equal(result$status, "success")
})

test_that("solve_with_ryacas returns failed status for garbage input", {
  result <- solve_with_ryacas("\\completely_invalid{{{")
  expect_equal(result$status, "failed")
})
```

- [ ] **Step 2: Implement `backend/plumber.R`**

```r
# Plumber API exposing Ryacas as a cross-check symbolic solver.
# Run via: Rscript run_plumber.r  -> http://127.0.0.1:8003

library(plumber)
library(Ryacas)
library(jsonlite)

# ---- LaTeX -> Yacas string conversion ------------------------------------

latex_to_yacas <- function(latex) {
  s <- latex
  # \frac{a}{b} -> (a)/(b)
  s <- gsub("\\\\frac\\{([^{}]*)\\}\\{([^{}]*)\\}", "((\\1)/(\\2))", s, perl = TRUE)
  # \sqrt{a} -> Sqrt(a)
  s <- gsub("\\\\sqrt\\{([^{}]*)\\}", "Sqrt(\\1)", s, perl = TRUE)
  # \cdot, \times -> *
  s <- gsub("\\\\(cdot|times)", "*", s, perl = TRUE)
  # \div -> /
  s <- gsub("\\\\div", "/", s, perl = TRUE)
  # \pi, \infty
  s <- gsub("\\\\pi", "Pi", s, perl = TRUE)
  s <- gsub("\\\\infty", "Infinity", s, perl = TRUE)
  # strip remaining backslashes from generic functions/letters (\sin, \alpha, ...)
  s <- gsub("\\\\([a-zA-Z]+)", "\\1", s, perl = TRUE)
  # remove all braces (we've already handled the structured ones above)
  s <- gsub("\\{|\\}", "", s, perl = TRUE)
  # implicit multiplication: 2x -> 2*x
  s <- gsub("([0-9])([a-zA-Z(])", "\\1*\\2", s, perl = TRUE)
  trimws(s)
}

detect_operation <- function(latex) {
  if (grepl("(?<![<>!])=(?!=)", latex, perl = TRUE)) return("solve")
  if (grepl("\\\\int", latex)) return("integrate")
  if (grepl("\\\\frac\\s*\\{d", latex) || grepl("'", latex) || grepl("\\\\prime", latex)) {
    return("differentiate")
  }
  if (grepl("\\\\lim", latex)) return("limit")
  "simplify"
}

# ---- per-operation solvers -----------------------------------------------

run_yacas <- function(expr_str) {
  trimws(as.character(yac(expr_str)))
}

solve_with_ryacas <- function(latex) {
  tryCatch({
    op <- detect_operation(latex)
    yacas_expr <- latex_to_yacas(latex)
    if (op == "solve") {
      sides <- strsplit(yacas_expr, "=", fixed = TRUE)[[1]]
      lhs <- sides[1]; rhs <- sides[2]
      sol <- run_yacas(sprintf("Solve(%s == %s, x)", lhs, rhs))
    } else if (op == "integrate") {
      body <- gsub("int", "", yacas_expr, fixed = TRUE)
      body <- trimws(gsub("dx$", "", body))
      sol <- run_yacas(sprintf("Integrate(x) %s", body))
    } else if (op == "differentiate") {
      body <- gsub("frac\\{d\\}\\{dx\\}", "", yacas_expr, perl = TRUE)
      body <- gsub("d/dx", "", body, fixed = TRUE)
      sol <- run_yacas(sprintf("D(x) %s", trimws(body)))
    } else if (op == "limit") {
      sol <- run_yacas(sprintf("Limit(x, 0) %s", yacas_expr))
    } else {
      sol <- run_yacas(sprintf("Simplify(%s)", yacas_expr))
    }
    list(
      status = "success",
      operation = op,
      solution = sol,
      latex_result = sol  # Yacas output is plain text; mathjax-renderable as-is in most cases
    )
  }, error = function(e) {
    list(status = "failed", error = conditionMessage(e))
  })
}

# ---- HTTP endpoints ------------------------------------------------------

#* Health probe
#* @get /health
function() {
  list(status = "ok", solver = "ryacas")
}

#* Solve an expression
#* @get /solve_ryacas
#* @param latex character The LaTeX expression
function(latex = "") {
  if (nchar(latex) == 0) {
    return(list(status = "failed", error = "Empty latex"))
  }
  solve_with_ryacas(latex)
}
```

- [ ] **Step 3: Implement `backend/run_plumber.r`**

```r
# Launch the Plumber API on port 8003.
library(plumber)
pr <- plumb("plumber.R")
pr$run(host = "127.0.0.1", port = 8003)
```

- [ ] **Step 4: Run R tests, expect pass**

From `backend/`:
```
Rscript -e "testthat::test_file('tests/test_plumber.R')"
```
Expected: all tests pass. If R packages are missing, install:
```
Rscript -e "install.packages(c('plumber','Ryacas','jsonlite','testthat'), repos='https://cloud.r-project.org')"
```

- [ ] **Step 5: Smoke-test the Plumber server manually**

In one terminal:
```
cd backend && Rscript run_plumber.r
```
In another:
```
curl "http://127.0.0.1:8003/health"
curl --get --data-urlencode "latex=x + 5 = 12" "http://127.0.0.1:8003/solve_ryacas"
```
Expected first: `{"status":["ok"], ...}`. Expected second: a JSON object with `"status":["success"]` and a solution mentioning `7`.

Stop the Plumber server with Ctrl+C.

- [ ] **Step 6: Commit**

```
git add backend/plumber.R backend/run_plumber.r backend/tests/test_plumber.R
git commit -m "feat: R Plumber API for Ryacas cross-solver on port 8003"
```

---

## Task 7: Python Ryacas client

**Files:**
- Create: `backend/ryacas_client.py`
- Create: `backend/tests/test_ryacas_client.py`

- [ ] **Step 1: Write failing tests** at `backend/tests/test_ryacas_client.py`

```python
import time
from unittest.mock import MagicMock, patch

import requests

from ryacas_client import RyacasResult, cross_solve, _health_cache


def setup_function(_):
    # reset health cache between tests
    _health_cache.clear()


def test_returns_none_when_unreachable():
    with patch("ryacas_client.requests.get", side_effect=requests.ConnectionError):
        assert cross_solve("x + 1") is None


def test_returns_none_on_timeout():
    with patch("ryacas_client.requests.get", side_effect=requests.Timeout):
        assert cross_solve("x + 1") is None


def test_returns_failed_on_5xx():
    health_resp = MagicMock(status_code=200, json=lambda: {"status": ["ok"]})
    bad_resp = MagicMock(status_code=500, text="boom")
    with patch("ryacas_client.requests.get", side_effect=[health_resp, bad_resp]):
        r = cross_solve("x + 1")
    assert r is not None
    assert r.status == "failed"
    assert "500" in (r.error or "")


def test_returns_success_on_valid_response():
    health_resp = MagicMock(status_code=200, json=lambda: {"status": ["ok"]})
    ok_resp = MagicMock(
        status_code=200,
        json=lambda: {"status": ["success"], "operation": ["solve"],
                      "solution": ["7"], "latex_result": ["7"]},
    )
    with patch("ryacas_client.requests.get", side_effect=[health_resp, ok_resp]):
        r = cross_solve("x + 5 = 12")
    assert r is not None
    assert r.status == "success"
    assert r.solution == "7"
    assert r.latex_result == "7"


def test_health_cache_avoids_repeated_probes():
    health_resp = MagicMock(status_code=200, json=lambda: {"status": ["ok"]})
    ok_resp = MagicMock(
        status_code=200,
        json=lambda: {"status": ["success"], "operation": ["simplify"],
                      "solution": ["x"], "latex_result": ["x"]},
    )
    with patch("ryacas_client.requests.get", side_effect=[health_resp, ok_resp, ok_resp]) as m:
        cross_solve("x")
        cross_solve("x")
    # Two cross_solve calls should produce 3 GETs (1 health + 2 solve), not 4.
    assert m.call_count == 3
```

- [ ] **Step 2: Run tests, expect failure**

```
.venv/Scripts/python -m pytest tests/test_ryacas_client.py -v
```
Expected: ImportError for `ryacas_client`.

- [ ] **Step 3: Implement `backend/ryacas_client.py`**

```python
"""HTTP client for the R Plumber cross-solver. Never raises."""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)

PLUMBER_URL = "http://127.0.0.1:8003"
DEFAULT_TIMEOUT_S = 3.0
HEALTH_CACHE_TTL_S = 30.0

_health_cache: dict[str, float] = {}  # {"last_ok": epoch_seconds}


@dataclass
class RyacasResult:
    status: str                # "success" | "failed"
    solution: Optional[str]
    latex_result: Optional[str]
    error: Optional[str]


def _is_healthy(timeout_s: float) -> bool:
    last_ok = _health_cache.get("last_ok", 0.0)
    if time.time() - last_ok < HEALTH_CACHE_TTL_S:
        return True
    try:
        r = requests.get(f"{PLUMBER_URL}/health", timeout=timeout_s)
        if r.status_code == 200:
            _health_cache["last_ok"] = time.time()
            return True
        return False
    except (requests.ConnectionError, requests.Timeout):
        return False
    except Exception as exc:
        logger.warning("Plumber health probe failed: %s", exc)
        return False


def _unwrap(field):
    """Plumber returns single-element JSON arrays for scalars; unwrap them."""
    if isinstance(field, list) and len(field) == 1:
        return field[0]
    return field


def cross_solve(latex: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> Optional[RyacasResult]:
    """Send latex to /solve_ryacas. Returns None if Plumber is unreachable."""
    if not _is_healthy(timeout_s=timeout_s):
        return None
    try:
        r = requests.get(
            f"{PLUMBER_URL}/solve_ryacas",
            params={"latex": latex},
            timeout=timeout_s,
        )
    except (requests.ConnectionError, requests.Timeout):
        return None
    except Exception as exc:
        logger.warning("Plumber request failed: %s", exc)
        return None

    if r.status_code != 200:
        return RyacasResult(
            status="failed",
            solution=None,
            latex_result=None,
            error=f"HTTP {r.status_code}: {r.text[:200]}",
        )
    try:
        data = r.json()
    except Exception:
        return RyacasResult(status="failed", solution=None, latex_result=None,
                            error="invalid JSON from Plumber")

    status = _unwrap(data.get("status"))
    if status == "success":
        return RyacasResult(
            status="success",
            solution=_unwrap(data.get("solution")),
            latex_result=_unwrap(data.get("latex_result")),
            error=None,
        )
    return RyacasResult(
        status="failed",
        solution=None,
        latex_result=None,
        error=str(_unwrap(data.get("error")) or "unknown ryacas error"),
    )
```

- [ ] **Step 4: Run tests, expect pass**

```
.venv/Scripts/python -m pytest tests/test_ryacas_client.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```
git add backend/ryacas_client.py backend/tests/test_ryacas_client.py
git commit -m "feat: Python client for Plumber cross-solver, with health cache"
```

---

## Task 8: Wire parallel solve + activity logging into /convert

**Files:**
- Modify: `backend/main.py`
- Modify: `backend/tests/test_main.py`

- [ ] **Step 1: Append failing tests** to `backend/tests/test_main.py`

Add to the bottom of the file:
```python
from ryacas_client import RyacasResult


def _accepted():
    from ml.classifier import ClassifyResult
    return ClassifyResult(predicted_latex="x", confidence=0.93, num_components=1, accepted=True)


def test_convert_includes_ryacas_when_available():
    """When Plumber is reachable and agrees with SymPy, response has agreement='match'."""
    mock_response = MagicMock()
    mock_response.text = "x + 5 = 12"

    sympy_result = {
        "status": "success", "operation": "solve", "operation_label": "Solving equation",
        "steps": [], "solution": "7", "latex_result": "7",
    }
    ryacas_result = RyacasResult(status="success", solution="7", latex_result="7", error=None)

    class FakeClassifier:
        def is_loaded(self): return False
        def classify(self, _): return None

    with patch.object(main, "classifier", FakeClassifier()), \
         patch.object(main._client.models, "generate_content", return_value=mock_response), \
         patch.object(main.solver, "solve_expression", return_value=sympy_result), \
         patch.object(main.ryacas_client, "cross_solve", return_value=ryacas_result):
        resp = _client.post("/convert", files={"file": ("b.png", _png_bytes(), "image/png")})

    data = resp.json()
    sd = data["solution_data"]
    assert sd["ryacas"]["latex_result"] == "7"
    assert sd["agreement"] == "match"


def test_convert_marks_unavailable_when_plumber_down():
    mock_response = MagicMock()
    mock_response.text = "x"
    sympy_result = {
        "status": "success", "operation": "simplify", "operation_label": "Simplifying",
        "steps": [], "solution": "x", "latex_result": "x",
    }

    class FakeClassifier:
        def is_loaded(self): return False
        def classify(self, _): return None

    with patch.object(main, "classifier", FakeClassifier()), \
         patch.object(main._client.models, "generate_content", return_value=mock_response), \
         patch.object(main.solver, "solve_expression", return_value=sympy_result), \
         patch.object(main.ryacas_client, "cross_solve", return_value=None):
        resp = _client.post("/convert", files={"file": ("b.png", _png_bytes(), "image/png")})

    sd = resp.json()["solution_data"]
    assert sd["ryacas"] is None
    assert sd["agreement"] == "ryacas_unavailable"


def test_convert_marks_differ_when_solvers_disagree():
    mock_response = MagicMock()
    mock_response.text = "x + 5 = 12"
    sympy_result = {
        "status": "success", "operation": "solve", "operation_label": "Solving equation",
        "steps": [], "solution": "7", "latex_result": "7",
    }
    ryacas_result = RyacasResult(status="success", solution="6", latex_result="6", error=None)

    class FakeClassifier:
        def is_loaded(self): return False
        def classify(self, _): return None

    with patch.object(main, "classifier", FakeClassifier()), \
         patch.object(main._client.models, "generate_content", return_value=mock_response), \
         patch.object(main.solver, "solve_expression", return_value=sympy_result), \
         patch.object(main.ryacas_client, "cross_solve", return_value=ryacas_result):
        resp = _client.post("/convert", files={"file": ("b.png", _png_bytes(), "image/png")})

    sd = resp.json()["solution_data"]
    assert sd["agreement"] == "differ"
    assert sd["ryacas"]["latex_result"] == "6"


def test_convert_logs_activity_row():
    mock_response = MagicMock()
    mock_response.text = "x"
    sympy_result = {
        "status": "success", "operation": "simplify", "operation_label": "Simplifying",
        "steps": [], "solution": "x", "latex_result": "x",
    }

    class FakeClassifier:
        def is_loaded(self): return False
        def classify(self, _): return None

    with patch.object(main, "classifier", FakeClassifier()), \
         patch.object(main._client.models, "generate_content", return_value=mock_response), \
         patch.object(main.solver, "solve_expression", return_value=sympy_result), \
         patch.object(main.ryacas_client, "cross_solve", return_value=None), \
         patch.object(main.activity_log, "log_request") as log_mock:
        _client.post("/convert", files={"file": ("b.png", _png_bytes(), "image/png")})

    assert log_mock.call_count == 1
    kwargs = log_mock.call_args.kwargs
    assert kwargs["source"] == "gemini"
    assert kwargs["recognized_latex"] == "x"
    assert kwargs["agreement"] == "ryacas_unavailable"
```

- [ ] **Step 2: Run tests, expect failure**

```
.venv/Scripts/python -m pytest tests/test_main.py -v
```
Expected: 4 failures (no `ryacas_client` attribute on main, no `activity_log`, no agreement field).

- [ ] **Step 3: Modify `backend/main.py`**

Replace the file contents with the version below. Key additions: parallel SymPy + Ryacas via ThreadPoolExecutor; activity logging at the end of `/convert`; new helper `_compute_agreement`.

```python
"""
MathBoard FastAPI backend.

Tries a local SymbolClassifier first; falls back to Google Gemini if the
classifier is disabled, indecisive, or sees a multi-symbol input. The
recognized LaTeX is then solved by SymPy and (in parallel) cross-checked by
Ryacas via the R Plumber API. Each request is appended to activity.db.
"""

import concurrent.futures
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

import activity_log
import ryacas_client
import solver
from ml.classifier import SymbolClassifier

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

if not GEMINI_API_KEY:
    print(
        "\n================================================================\n"
        "  WARNING: GEMINI_API_KEY is not set.\n"
        "  Create backend/.env with GEMINI_API_KEY=your_key_here\n"
        "  Generate a key at https://aistudio.google.com\n"
        "================================================================\n",
        file=sys.stderr, flush=True,
    )
    _client = None
else:
    _client = genai.Client(api_key=GEMINI_API_KEY)

_BACKEND_DIR = Path(__file__).parent
_ARTIFACTS = _BACKEND_DIR / "ml" / "artifacts"
_ACTIVITY_DB = _BACKEND_DIR / "activity.db"

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
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
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
        m = re.search(r"https://console\.developers\.google\.com/[^\s'\"]+", msg)
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


def _normalize_for_compare(s: str | None) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", "", s)


def _compute_agreement(sympy_latex: str | None, ryacas_result) -> str:
    if ryacas_result is None:
        return "ryacas_unavailable"
    if ryacas_result.status != "success":
        return "ryacas_error"
    if _normalize_for_compare(sympy_latex) == _normalize_for_compare(ryacas_result.latex_result):
        return "match"
    return "differ"


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
    t0 = time.monotonic()
    try:
        image_bytes = await file.read()
    except Exception as exc:
        return {"error": f"Failed to read uploaded file: {exc}"}
    if not image_bytes:
        return {"error": "Uploaded image was empty."}

    # ---- OCR (local first, gemini fallback) -------------------------------
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
            return {"error": "Gemini returned an empty response. Try drawing a clearer expression."}
        source = "gemini"
        confidence = None
        print(f"[Gemini Result] {latex_string}", flush=True)

    # ---- Solve in parallel: SymPy + Ryacas --------------------------------
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        sympy_future = ex.submit(solver.solve_expression, latex_string)
        ryacas_future = ex.submit(ryacas_client.cross_solve, latex_string)
        solution_data = sympy_future.result()
        ryacas_result = ryacas_future.result()

    solution_data["ryacas"] = (
        None if ryacas_result is None
        else {
            "status": ryacas_result.status,
            "solution": ryacas_result.solution,
            "latex_result": ryacas_result.latex_result,
        }
    )
    solution_data["agreement"] = _compute_agreement(
        solution_data.get("latex_result"), ryacas_result
    )

    duration_ms = int((time.monotonic() - t0) * 1000)

    activity_log.log_request(
        db_path=_ACTIVITY_DB,
        source=source,
        recognized_latex=latex_string,
        confidence=confidence,
        num_components=local_result.num_components if local_result else None,
        operation=solution_data.get("operation"),
        sympy_solution=solution_data.get("latex_result"),
        ryacas_solution=(solution_data["ryacas"] or {}).get("latex_result"),
        agreement=solution_data["agreement"],
        duration_ms=duration_ms,
        image_bytes=image_bytes,
    )

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
.venv/Scripts/python -m pytest tests/ 2>&1 | tail -8
```
Expected: 4 new test_main tests pass. Existing test_main tests still pass (their patches don't touch ryacas_client and the new code path returns `agreement: "ryacas_unavailable"` when not patched, which old tests don't assert against).

- [ ] **Step 5: Commit**

```
git add backend/main.py backend/tests/test_main.py
git commit -m "feat: parallel SymPy+Ryacas in /convert; log every request to activity.db"
```

---

## Task 9: Frontend — agreement line + tab label

**Files:**
- Modify: `frontend/mathboard/src/App.jsx`
- Modify: `frontend/mathboard/src/App.css`

- [ ] **Step 1: Extend `convertResult`** in `App.jsx`

Find the `setConvertResult({ ... })` block in `handleConvert` (currently includes `source` and `confidence`). Append two more fields:
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
  agreement: solData.agreement || "ryacas_unavailable",
  ryacasLatex: (solData.ryacas && solData.ryacas.latex_result) || "",
});
```

- [ ] **Step 2: Render the agreement line** in `App.jsx`

Find the JSX that renders the solution box (the block that starts with `<h3 className="result-label">Solution</h3>`). Just BELOW the solution box closing `</div>` (still inside the `result-success` block), add:
```jsx
{convertResult.agreement === "match" && (
  <div className="agreement agreement-match">
    ✓ Cross-checked with Ryacas (results agree)
  </div>
)}
{convertResult.agreement === "differ" && (
  <div className="agreement agreement-differ">
    ⚠ Ryacas got a different answer:&nbsp;
    <span
      dangerouslySetInnerHTML={{
        __html: katex.renderToString(convertResult.ryacasLatex || "", {
          throwOnError: false,
        }),
      }}
    />
  </div>
)}
```

(Do NOT render anything for `ryacas_unavailable` or `ryacas_error` — silent on absence.)

- [ ] **Step 3: Rename the tab** in `App.jsx`

Find the line that reads `Dataset Explorer` (the tab label). Change to `Model & Activity`. Find the `aria-label` attribute (if any) and update similarly. Find the iframe `title` and change to `MathBoard Model & Activity Dashboard`.

- [ ] **Step 4: Append CSS** to `App.css`

Append to the bottom of the file:
```css
/* ---------- Ryacas agreement line ---------- */
.agreement {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  border-radius: 8px;
  font-size: 0.85rem;
  font-weight: 600;
  margin-top: 4px;
  align-self: flex-start;
}

.agreement-match {
  color: #15803d;
  background: rgba(34, 197, 94, 0.10);
  border: 1px solid rgba(34, 197, 94, 0.35);
}

.agreement-differ {
  color: #b45309;
  background: rgba(245, 158, 11, 0.10);
  border: 1px solid rgba(245, 158, 11, 0.45);
}

[data-theme="dark"] .agreement-match {
  background: rgba(34, 197, 94, 0.18);
}

[data-theme="dark"] .agreement-differ {
  background: rgba(245, 158, 11, 0.20);
}
```

- [ ] **Step 5: Verify lint + build**

From `frontend/mathboard/`:
```
npm run lint && npm run build
```
Expected: no errors, no new warnings.

- [ ] **Step 6: Commit**

```
git add frontend/mathboard/src/App.jsx frontend/mathboard/src/App.css
git commit -m "feat(ui): show Ryacas agreement line; rename tab to Model & Activity"
```

---

## Task 10: Shiny dashboard rewrite

**Files:**
- Modify (full rewrite): `backend/shiny_app.R`

- [ ] **Step 1: Rewrite `backend/shiny_app.R`**

Replace the file contents entirely:
```r
# MathBoard "Model & Activity" Shiny dashboard.
# Reads:
#   backend/activity.db                — live request log
#   backend/ml/artifacts/metrics.json  — training-time metrics
#   backend/ml/artifacts/confusion.json — training-time confusion matrix
#
# Run via: Rscript -e "shiny::runApp('shiny_app.R', host='0.0.0.0', port=3838, launch.browser=FALSE)"

library(shiny)
library(shinydashboard)
library(ggplot2)
library(dplyr)
library(DBI)
library(RSQLite)
library(jsonlite)

ACTIVITY_DB     <- "activity.db"
METRICS_JSON    <- "ml/artifacts/metrics.json"
CONFUSION_JSON  <- "ml/artifacts/confusion.json"
REFRESH_MS      <- 5000

read_activity <- function(limit = 50) {
  if (!file.exists(ACTIVITY_DB)) return(data.frame())
  con <- tryCatch(dbConnect(SQLite(), ACTIVITY_DB), error = function(e) NULL)
  if (is.null(con)) return(data.frame())
  on.exit(dbDisconnect(con))
  tryCatch(
    dbGetQuery(con, "SELECT * FROM requests ORDER BY id DESC LIMIT ?", params = list(limit)),
    error = function(e) data.frame()
  )
}

read_activity_all <- function() {
  if (!file.exists(ACTIVITY_DB)) return(data.frame())
  con <- tryCatch(dbConnect(SQLite(), ACTIVITY_DB), error = function(e) NULL)
  if (is.null(con)) return(data.frame())
  on.exit(dbDisconnect(con))
  tryCatch(dbGetQuery(con, "SELECT * FROM requests"), error = function(e) data.frame())
}

read_metrics <- function() {
  if (!file.exists(METRICS_JSON)) return(NULL)
  tryCatch(jsonlite::fromJSON(METRICS_JSON), error = function(e) NULL)
}

read_confusion <- function() {
  if (!file.exists(CONFUSION_JSON)) return(NULL)
  tryCatch(jsonlite::fromJSON(CONFUSION_JSON), error = function(e) NULL)
}

ui <- dashboardPage(
  skin = "blue",
  dashboardHeader(title = "MathBoard — Model & Activity"),
  dashboardSidebar(disable = TRUE),
  dashboardBody(
    tags$head(tags$style(HTML(
      ".content-wrapper { background: #f4f6f9; }
       .small-box .icon-large { right: 14px; }
       .badge-local { color: #4f46e5; font-weight: 700; }
       .badge-gemini { color: #64748b; font-weight: 700; }
       .agree-match { color: #15803d; font-weight: 700; }
       .agree-differ { color: #b45309; font-weight: 700; }
       .agree-na { color: #94a3b8; }
       .thumb { width: 48px; height: 48px; border: 1px solid #e6e8ef; border-radius: 4px; }"
    ))),
    fluidRow(
      valueBoxOutput("card_local_rate"),
      valueBoxOutput("card_accepted_acc"),
      valueBoxOutput("card_agreement"),
      valueBoxOutput("card_total")
    ),
    tabsetPanel(
      id = "main_tabs",
      tabPanel("Recent Activity", uiOutput("recent_table_ui")),
      tabPanel("Model Performance",
        fluidRow(
          box(title = "Confusion Matrix", width = 12, status = "primary",
              plotOutput("confusion_plot", height = "560px"))
        ),
        fluidRow(
          box(title = "Per-class accuracy (top 30)", width = 6, status = "info",
              plotOutput("perclass_plot", height = "420px")),
          box(title = "Confidence histogram (local model only)", width = 6, status = "info",
              plotOutput("conf_hist", height = "420px"))
        )
      ),
      tabPanel("Solver Agreement",
        fluidRow(
          box(title = "Daily agreement breakdown", width = 12, status = "warning",
              plotOutput("agree_over_time", height = "320px"))
        ),
        fluidRow(
          box(title = "Disagreements (SymPy ≠ Ryacas)", width = 12, status = "danger",
              uiOutput("disagree_table_ui"))
        )
      ),
      tabPanel("Local vs Gemini",
        fluidRow(
          box(title = "Daily volume", width = 12, status = "primary",
              plotOutput("volume_plot", height = "320px"))
        ),
        fluidRow(
          box(title = "Top 10 handled locally", width = 6, status = "success",
              plotOutput("top_local", height = "320px")),
          box(title = "Top 10 falling through to Gemini", width = 6, status = "warning",
              plotOutput("top_gemini", height = "320px"))
        )
      )
    )
  )
)

server <- function(input, output, session) {
  tick <- reactiveTimer(REFRESH_MS)

  activity <- reactive({ tick(); read_activity_all() })
  recent <- reactive({ tick(); read_activity(limit = 50) })
  metrics <- reactive({ tick(); read_metrics() })
  confusion <- reactive({ tick(); read_confusion() })

  empty_card <- function(title) {
    valueBox("—", title, icon = icon("hourglass-half"), color = "light-blue")
  }

  output$card_local_rate <- renderValueBox({
    df <- activity()
    if (nrow(df) == 0) return(empty_card("Local hit rate"))
    rate <- mean(df$source == "local") * 100
    valueBox(sprintf("%.1f%%", rate), "Local hit rate",
             icon = icon("bolt"), color = "purple")
  })

  output$card_accepted_acc <- renderValueBox({
    m <- metrics()
    if (is.null(m)) return(empty_card("Accuracy on accepted"))
    valueBox(sprintf("%.1f%%", m$accuracy_on_accepted * 100),
             "Accuracy on accepted",
             icon = icon("bullseye"), color = "green")
  })

  output$card_agreement <- renderValueBox({
    df <- activity()
    if (nrow(df) == 0) return(empty_card("Solver agreement"))
    rate <- mean(df$agreement == "match", na.rm = TRUE) * 100
    valueBox(sprintf("%.1f%%", rate), "Solver agreement",
             icon = icon("check-double"), color = "yellow")
  })

  output$card_total <- renderValueBox({
    df <- activity()
    valueBox(format(nrow(df), big.mark = ","), "Total requests logged",
             icon = icon("list"), color = "blue")
  })

  # ---- Tab 1: Recent Activity --------------------------------------------
  output$recent_table_ui <- renderUI({
    df <- recent()
    if (nrow(df) == 0) {
      return(div(style = "padding: 20px; color: #64748b;",
                 "No requests logged yet. Use the Math Solver tab to generate data."))
    }
    rows <- lapply(seq_len(nrow(df)), function(i) {
      r <- df[i, ]
      thumb_src <- if (!is.na(r$thumbnail_b64)) {
        sprintf("data:image/png;base64,%s", r$thumbnail_b64)
      } else "data:,"
      src_class <- if (!is.na(r$source) && r$source == "local") "badge-local" else "badge-gemini"
      agree_class <- switch(as.character(r$agreement),
                            "match" = "agree-match",
                            "differ" = "agree-differ",
                            "agree-na")
      tags$tr(
        tags$td(format(as.POSIXct(r$timestamp, tz = "UTC"), "%H:%M:%S")),
        tags$td(tags$img(src = thumb_src, class = "thumb")),
        tags$td(tags$code(r$recognized_latex)),
        tags$td(class = src_class, toupper(r$source)),
        tags$td(if (!is.na(r$confidence)) sprintf("%.2f", r$confidence) else "—"),
        tags$td(if (!is.na(r$sympy_solution)) tags$code(r$sympy_solution) else "—"),
        tags$td(if (!is.na(r$ryacas_solution)) tags$code(r$ryacas_solution) else "—"),
        tags$td(class = agree_class, r$agreement)
      )
    })
    tags$table(
      class = "table table-striped table-condensed",
      tags$thead(tags$tr(
        tags$th("Time"), tags$th("Thumb"), tags$th("Recognized"),
        tags$th("Source"), tags$th("Conf"),
        tags$th("SymPy"), tags$th("Ryacas"), tags$th("Agreement")
      )),
      tags$tbody(rows)
    )
  })

  # ---- Tab 2: Model Performance ------------------------------------------
  output$confusion_plot <- renderPlot({
    cf <- confusion()
    if (is.null(cf)) return(plot.new())
    classes <- unlist(cf$classes)
    mat <- as.matrix(cf$matrix)
    df <- expand.grid(true = classes, pred = classes, stringsAsFactors = FALSE)
    df$count <- as.vector(mat)
    df_top <- df %>% group_by(true) %>% summarise(total = sum(count)) %>%
      arrange(desc(total)) %>% head(30)
    df2 <- df %>% filter(true %in% df_top$true, pred %in% df_top$true)
    df2$true <- factor(df2$true, levels = df_top$true)
    df2$pred <- factor(df2$pred, levels = df_top$true)
    ggplot(df2, aes(pred, true, fill = count)) +
      geom_tile() +
      scale_fill_gradient(low = "white", high = "#4f46e5") +
      theme_minimal(base_size = 11) +
      theme(axis.text.x = element_text(angle = 60, hjust = 1)) +
      labs(x = "Predicted", y = "True", fill = "Count")
  })

  output$perclass_plot <- renderPlot({
    cf <- confusion()
    if (is.null(cf)) return(plot.new())
    classes <- unlist(cf$classes)
    mat <- as.matrix(cf$matrix)
    diag_vals <- diag(mat)
    totals <- rowSums(mat)
    acc <- ifelse(totals == 0, 0, diag_vals / totals)
    df <- data.frame(class = classes, accuracy = acc, count = totals) %>%
      arrange(desc(count)) %>% head(30)
    df$class <- factor(df$class, levels = rev(df$class))
    ggplot(df, aes(class, accuracy)) +
      geom_col(fill = "#4f46e5") +
      coord_flip() + theme_minimal(base_size = 11) +
      labs(x = NULL, y = "Per-class accuracy") +
      ylim(0, 1)
  })

  output$conf_hist <- renderPlot({
    df <- activity() %>% filter(source == "local", !is.na(confidence))
    if (nrow(df) == 0) {
      return(plot.new())
    }
    ggplot(df, aes(confidence)) +
      geom_histogram(binwidth = 0.05, fill = "#4f46e5", boundary = 0) +
      scale_x_continuous(limits = c(0, 1)) + theme_minimal(base_size = 11) +
      labs(x = "Confidence", y = "Count")
  })

  # ---- Tab 3: Solver agreement -------------------------------------------
  output$agree_over_time <- renderPlot({
    df <- activity()
    if (nrow(df) == 0) return(plot.new())
    df$date <- as.Date(df$timestamp)
    daily <- df %>% group_by(date, agreement) %>% summarise(n = n(), .groups = "drop")
    ggplot(daily, aes(date, n, fill = agreement)) +
      geom_bar(stat = "identity") +
      scale_fill_manual(values = c(
        match = "#15803d", differ = "#b45309",
        ryacas_unavailable = "#94a3b8", ryacas_error = "#dc2626"
      )) +
      theme_minimal(base_size = 12) + labs(x = NULL, y = "Requests", fill = "Agreement")
  })

  output$disagree_table_ui <- renderUI({
    df <- activity() %>% filter(agreement == "differ") %>% arrange(desc(timestamp)) %>% head(50)
    if (nrow(df) == 0) {
      return(div(style = "padding: 14px; color: #64748b;",
                 "No disagreements logged yet."))
    }
    tags$table(class = "table table-striped table-condensed",
      tags$thead(tags$tr(
        tags$th("Time"), tags$th("Recognized"),
        tags$th("SymPy"), tags$th("Ryacas")
      )),
      tags$tbody(lapply(seq_len(nrow(df)), function(i) {
        r <- df[i, ]
        tags$tr(
          tags$td(format(as.POSIXct(r$timestamp, tz = "UTC"), "%H:%M:%S")),
          tags$td(tags$code(r$recognized_latex)),
          tags$td(tags$code(r$sympy_solution)),
          tags$td(tags$code(r$ryacas_solution))
        )
      }))
    )
  })

  # ---- Tab 4: Local vs Gemini --------------------------------------------
  output$volume_plot <- renderPlot({
    df <- activity()
    if (nrow(df) == 0) return(plot.new())
    df$date <- as.Date(df$timestamp)
    daily <- df %>% group_by(date, source) %>% summarise(n = n(), .groups = "drop")
    ggplot(daily, aes(date, n, color = source)) +
      geom_line(linewidth = 1) + geom_point(size = 2) +
      scale_color_manual(values = c(local = "#4f46e5", gemini = "#64748b")) +
      theme_minimal(base_size = 12) + labs(x = NULL, y = "Requests", color = "Source")
  })

  output$top_local <- renderPlot({
    df <- activity() %>% filter(source == "local")
    if (nrow(df) == 0) return(plot.new())
    top <- df %>% count(recognized_latex, sort = TRUE) %>% head(10)
    top$recognized_latex <- factor(top$recognized_latex, levels = rev(top$recognized_latex))
    ggplot(top, aes(recognized_latex, n)) +
      geom_col(fill = "#15803d") + coord_flip() + theme_minimal(base_size = 11) +
      labs(x = NULL, y = "Count")
  })

  output$top_gemini <- renderPlot({
    df <- activity() %>% filter(source == "gemini")
    if (nrow(df) == 0) return(plot.new())
    top <- df %>% count(recognized_latex, sort = TRUE) %>% head(10)
    top$recognized_latex <- factor(top$recognized_latex, levels = rev(top$recognized_latex))
    ggplot(top, aes(recognized_latex, n)) +
      geom_col(fill = "#b45309") + coord_flip() + theme_minimal(base_size = 11) +
      labs(x = NULL, y = "Count")
  })
}

shinyApp(ui, server, options = list(port = 3838, host = "0.0.0.0", launch.browser = FALSE))
```

- [ ] **Step 2: Smoke-test the dashboard manually**

In one terminal (with backend training already done so artifacts exist):
```
cd backend && Rscript -e "shiny::runApp('shiny_app.R', host='0.0.0.0', port=3838, launch.browser=FALSE)"
```
If R packages are missing:
```
Rscript -e "install.packages(c('shiny','shinydashboard','ggplot2','dplyr','DBI','RSQLite','jsonlite'), repos='https://cloud.r-project.org')"
```
Open http://localhost:3838 in a browser. With no `activity.db` yet, expect:
- Cards show "—" for activity-derived numbers, accuracy card shows the trained accuracy.
- Recent Activity tab shows "No requests logged yet."
- Model Performance tab shows confusion matrix + per-class accuracy plot.
- Solver Agreement / Local vs Gemini tabs show empty plots gracefully.

Stop with Ctrl+C.

- [ ] **Step 3: Commit**

```
git add backend/shiny_app.R
git commit -m "feat: rewrite Shiny dashboard as 4-tab Model & Activity view"
```

---

## Task 11: Documentation, run-everything check

**Files:**
- Modify: `SETUP.md`
- Modify: `run_shiny.r` (no change needed if it already calls shinyApp; just verify)

- [ ] **Step 1: Update `SETUP.md` to describe the new optional terminals**

Find the "Running the app" section. Replace the existing "(optional) Terminal 3" block with two optional terminals:

```markdown
**Terminal 3 (optional) — R Plumber cross-solver:**
```
cd backend
Rscript run_plumber.r
```
→ Runs at http://127.0.0.1:8003. When running, every solved expression is
cross-checked by Ryacas, and the UI shows "✓ Cross-checked" or "⚠ Differ".

**Terminal 4 (optional) — R Shiny dashboard ("Model & Activity" tab):**
```
cd backend
Rscript run_shiny.r
```
→ Runs at http://localhost:3838. Powers the React tab "Model & Activity" via iframe.

The Math Solver tab works without Terminals 3 or 4. Each is independently optional.
```

In the R-packages install snippet, replace the line with this expanded set:
```
install.packages(c(
  "shiny", "shinydashboard", "ggplot2", "dplyr",
  "plumber", "Ryacas", "jsonlite",
  "DBI", "RSQLite", "testthat"
))
```

- [ ] **Step 2: Verify `run_shiny.r` still works**

Open `backend/run_shiny.r`. Ensure it calls `shinyApp` from `shiny_app.R`. If not (e.g., if it explicitly references the old HASYv2 setup), replace its contents with:
```r
# Launch the Shiny dashboard on port 3838.
shiny::runApp("shiny_app.R", host = "0.0.0.0", port = 3838, launch.browser = FALSE)
```

- [ ] **Step 3: Run full backend test suite**

From `backend/`:
```
.venv/Scripts/python -m pytest tests/ 2>&1 | tail -5
```
Expected: all tests pass — tally is roughly 31 (existing) + 10 (classifier) + 1 (train smoke) + 3 (idx_parser) + 6 (datasets) + 6 (activity_log) + 5 (ryacas_client) + 4 (new test_main) = 66 tests.

- [ ] **Step 4: Frontend lint + build**

From `frontend/mathboard/`:
```
npm run lint && npm run build
```
Expected: 0 errors, 0 new warnings.

- [ ] **Step 5: End-to-end manual smoke test**

Start backend, frontend, Plumber, and Shiny in four terminals (per updated SETUP.md). In a browser:

1. Visit http://localhost:5173
2. Draw a single digit (e.g., `7`); click Convert.
3. Confirm result panel shows source badge (`⚡ Local model` or `☁ Gemini`) AND agreement line (✓ or ⚠ if Plumber is up; nothing if not).
4. Switch to "Model & Activity" tab. Confirm the iframe loads the dashboard. Confirm your last request shows up in the Recent Activity table.

- [ ] **Step 6: Commit**

```
git add SETUP.md backend/run_shiny.r
git commit -m "docs: SETUP.md covers Plumber + new Shiny dashboard terminals"
```

---

## Final verification

- [ ] **All backend tests green**: `pytest tests/` reports 0 failures.
- [ ] **R tests green**: `Rscript -e "testthat::test_file('tests/test_plumber.R')"` reports 0 failures.
- [ ] **Frontend builds clean**: `npm run lint && npm run build` no errors.
- [ ] **End-to-end works**: from a fresh restart, drawing a digit produces a valid response with source, agreement, and the dashboard reflects the request within 5 seconds.
