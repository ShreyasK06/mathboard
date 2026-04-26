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
