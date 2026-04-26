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


def test_hasy_dataset_loads_correct_shape(tmp_path):
    classes = ["A", "B"]
    _make_synthetic_hasy(tmp_path, classes, samples_per_class=3)
    rows = [(f"hasy-data/v2-{i:05d}.png", i % 2) for i in range(6)]
    ds = HasyDataset(rows, image_root=tmp_path, augment=False, mean=0.5, std=0.5)
    assert len(ds) == 6
    x, y = ds[0]
    assert x.shape == (1, 32, 32)
    assert isinstance(y, int)


def test_mnist_dataset_polarity_inverted_to_match_hasyv2():
    images, labels = _make_synthetic_mnist_arrays()
    ds = MnistDataset(images, labels, augment=False, mean=0.5, std=0.5, label_offset=100)
    x, _ = ds[0]
    middle = float(x[0, 12:20, 12:20].mean())
    edge = float(x[0, 0:4, 0:4].mean())
    assert middle < edge, f"middle should be darker than edge after inversion (got middle={middle}, edge={edge})"


def test_mnist_dataset_resized_to_32x32():
    images, labels = _make_synthetic_mnist_arrays()
    ds = MnistDataset(images, labels, augment=False, mean=0.5, std=0.5, label_offset=0)
    x, _ = ds[0]
    assert x.shape == (1, 32, 32)


def test_mnist_dataset_applies_label_offset():
    images, labels = _make_synthetic_mnist_arrays()
    ds = MnistDataset(images, labels, augment=False, mean=0.5, std=0.5, label_offset=100)
    seen = {ds[i][1] for i in range(len(ds))}
    assert seen == set(range(100, 110))


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
    assert any(s < 2 for s in seen)
    assert any(s >= 2 for s in seen)


def test_weighted_sampler_balances_classes():
    labels = [0] * 1000 + [1] * 100
    sampler = build_weighted_sampler(labels, num_samples=4400, seed=42)
    drawn = [labels[i] for i in sampler]
    c0 = sum(1 for l in drawn if l == 0)
    c1 = sum(1 for l in drawn if l == 1)
    assert 0.4 * len(drawn) < c0 < 0.6 * len(drawn)
    assert 0.4 * len(drawn) < c1 < 0.6 * len(drawn)
