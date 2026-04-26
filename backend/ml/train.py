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

    raw = _read_labels(data_root / "hasy-data-labels.csv")
    print(f"[train] HASYv2: {len(raw):,} rows", flush=True)
    hasy_classes = _pick_top_classes(raw, top_n_classes)
    cls_to_idx = {c: i for i, c in enumerate(hasy_classes)}
    hasy_rows = [(r["path"], cls_to_idx[r["latex"]]) for r in raw if r["latex"] in cls_to_idx]
    print(f"[train] HASYv2: top {len(hasy_classes)} classes, {len(hasy_rows):,} samples", flush=True)
    h_train, h_val = _stratified_split(hasy_rows, DEFAULT_VAL_FRAC, seed)

    classes = list(hasy_classes)
    label_offset = len(hasy_classes)
    m_train_imgs = m_train_lbls = m_val_imgs = m_val_lbls = None
    if include_mnist:
        mnist_dir = mnist_dir or (data_root.parent / "mnist")
        print("[train] loading MNIST...", flush=True)
        m_imgs, m_lbls = load_mnist(mnist_dir)
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
