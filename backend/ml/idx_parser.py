"""Minimal MNIST IDX file parser.

The IDX format header is:
- magic number (4 bytes, big-endian): 0x00000803 for images, 0x00000801 for labels
- count (4 bytes, big-endian)
- for images only: rows (4 bytes), cols (4 bytes)
- pixel/label bytes follow

Files may be gzipped (.gz) or raw.
"""

import gzip
import struct
from pathlib import Path

import numpy as np


def _open(path: Path):
    return gzip.open(path, "rb") if str(path).endswith(".gz") else open(path, "rb")


def read_idx_images(path: Path) -> np.ndarray:
    with _open(path) as f:
        magic, n, h, w = struct.unpack(">IIII", f.read(16))
        if magic != 0x00000803:
            raise ValueError(f"Not an IDX images file (magic={magic:#010x})")
        data = np.frombuffer(f.read(n * h * w), dtype=np.uint8)
        return data.reshape(n, h, w)


def read_idx_labels(path: Path) -> np.ndarray:
    with _open(path) as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 0x00000801:
            raise ValueError(f"Not an IDX labels file (magic={magic:#010x})")
        return np.frombuffer(f.read(n), dtype=np.uint8)
