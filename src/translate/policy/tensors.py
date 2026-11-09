"""Framework-free tensor record produced by the checkpoint reader."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

# dtype tag -> numpy little-endian dtype string. Only float32 is decoded today.
NUMPY_DTYPES = {"float32": "<f4"}


@dataclass(frozen=True)
class Tensor:
    """One state_dict entry: name, dtype tag, shape and raw little-endian bytes (C order)."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    data: bytes

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def sha256(self) -> str:
        """Hash of the raw bytes; equals ``sha256(t.numpy().tobytes())`` from torch on little-endian hosts."""
        return hashlib.sha256(self.data).hexdigest()

    def to_numpy(self) -> np.ndarray:
        """Read-only array view of the raw bytes."""
        array = np.frombuffer(self.data, dtype=NUMPY_DTYPES[self.dtype])
        return array.reshape(self.shape)


def total_params(tensors: tuple[Tensor, ...]) -> int:
    return sum(tensor.numel for tensor in tensors)
