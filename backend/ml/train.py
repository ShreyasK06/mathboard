"""Train a SymbolCNN on HASYv2 top-N classes.

Run from backend/:
    python -m ml.train

Reads data/hasy/hasy-data-labels.csv (+ data/hasy/hasy-data/*.png),
writes ml/artifacts/{model.pt, classes.json, metrics.json}.
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
        return t, label


def _read_labels(labels_csv: Path) -> list[dict]:
    with open(labels_csv, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pick_top_classes(rows: list[dict], top_n: int) -> list[str]:
    counts = Counter(r["latex"] for r in rows)
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
    for _label, items in by_label.items():
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
    best_metrics: dict = {}
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _train_one_epoch(model, train_loader, optim, criterion, device)
        val_loss, val_acc, coverage, acc_accepted = _evaluate(
            model, val_loader, criterion, device
        )
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
