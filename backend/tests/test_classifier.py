import torch

from ml.model import SymbolCNN


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
