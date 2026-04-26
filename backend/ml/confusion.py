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
