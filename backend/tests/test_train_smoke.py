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
