import gzip
import struct
from pathlib import Path

import numpy as np

from ml.idx_parser import read_idx_images, read_idx_labels


def _write_idx_images(path: Path, arr: np.ndarray) -> None:
    n, h, w = arr.shape
    header = struct.pack(">IIII", 0x00000803, n, h, w)
    with gzip.open(path, "wb") as f:
        f.write(header + arr.astype(np.uint8).tobytes())


def _write_idx_labels(path: Path, labels: np.ndarray) -> None:
    n = labels.shape[0]
    header = struct.pack(">II", 0x00000801, n)
    with gzip.open(path, "wb") as f:
        f.write(header + labels.astype(np.uint8).tobytes())


def test_read_idx_images_returns_correct_shape(tmp_path):
    arr = (np.random.rand(7, 28, 28) * 255).astype(np.uint8)
    p = tmp_path / "imgs.idx.gz"
    _write_idx_images(p, arr)
    out = read_idx_images(p)
    assert out.shape == (7, 28, 28)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, arr)


def test_read_idx_labels_returns_correct_shape(tmp_path):
    labels = np.array([0, 5, 9, 3, 1], dtype=np.uint8)
    p = tmp_path / "labels.idx.gz"
    _write_idx_labels(p, labels)
    out = read_idx_labels(p)
    assert out.shape == (5,)
    np.testing.assert_array_equal(out, labels)


def test_read_idx_images_handles_uncompressed(tmp_path):
    arr = (np.random.rand(3, 28, 28) * 255).astype(np.uint8)
    p = tmp_path / "imgs.idx"
    header = struct.pack(">IIII", 0x00000803, 3, 28, 28)
    p.write_bytes(header + arr.tobytes())
    out = read_idx_images(p)
    assert out.shape == (3, 28, 28)
