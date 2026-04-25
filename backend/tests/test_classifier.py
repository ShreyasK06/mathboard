import io

import numpy as np
import torch
from PIL import Image, ImageDraw

from ml.model import SymbolCNN
from ml.preprocessing import (
    binarize_and_count_components,
    crop_to_bbox,
    normalize,
    resize_to_32,
)


def _make_image(width: int, height: int, draw_fn) -> bytes:
    img = Image.new("L", (width, height), color=255)
    draw_fn(ImageDraw.Draw(img))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_symbol_cnn_forward_shape():
    """Forward pass returns logits of shape (batch, num_classes)."""
    model = SymbolCNN(num_classes=100)
    x = torch.zeros(2, 1, 32, 32)
    logits = model(x)
    assert logits.shape == (2, 100)


def test_symbol_cnn_param_count_is_small():
    """Model is intentionally small so CPU inference and training are fast.
    Threshold matches the architecture defined in model.py: two conv blocks
    feeding a 4096->128 FC bottleneck, ~600k params total."""
    model = SymbolCNN(num_classes=100)
    n = sum(p.numel() for p in model.parameters())
    assert n < 1_000_000, f"SymbolCNN has {n:,} params (expected <1M)"


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
    # After cropping to the bbox, the ink ends up roughly centered in the
    # 32x32 frame. Ink pixels normalize to +1.0, background to -1.0, so the
    # *positive* signal should be concentrated in the middle 16x16 region.
    middle_signal = float(arr[0, 0, 8:24, 8:24].clamp(min=0).sum())
    edge_signal = float(arr[0, 0].clamp(min=0).sum() - middle_signal)
    assert middle_signal > edge_signal
