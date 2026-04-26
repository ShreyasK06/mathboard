"""Dataset classes for HASYv2, MNIST, and the combined training set.

Both single-source datasets emit (1, 32, 32) float tensors with HASYv2's
polarity (ink dark on bright background). HASYv2 is loaded as-is; MNIST
is inverted (255 - pixel) and resized 28x28 -> 32x32 to match.
"""

import random
import urllib.request
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
        label_offset: int,
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


# ---- MNIST download ------------------------------------------------------

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
