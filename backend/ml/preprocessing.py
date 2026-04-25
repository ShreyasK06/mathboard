"""Shared preprocessing for both training and inference.

Both pipelines MUST use these functions to keep training and serving in sync.
"""

import io
from typing import Tuple

import numpy as np
import torch
from PIL import Image


def _otsu_threshold(gray: np.ndarray) -> int:
    """Standard Otsu threshold on a uint8 grayscale array.

    Returns t such that pixels with value <= t are class 0 (ink/dark) and
    pixels with value > t are class 1 (background/light). When variance ties
    across a plateau (common for bimodal images like ours), we take the last
    t in the plateau so the threshold sits between the modes, not at the
    start of class 0.
    """
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    total = gray.size
    sum_total = float((np.arange(256) * hist).sum())
    sum_b, w_b, max_var, threshold = 0.0, 0, -1.0, 0
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
        if var >= max_var:
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
    mask = arr <= threshold  # ink = at or below threshold (class 0)
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
