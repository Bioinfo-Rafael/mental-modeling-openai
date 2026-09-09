"""Raw access contracts and source provenance, independent of model preprocessing."""
from __future__ import annotations
from dataclasses import dataclass, field
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"


@lru_cache(maxsize=1)
def inventories():
    base = ROOT / "outputs/dataset_inventory"
    return (json.loads((base / "llmx_schema.json").read_text()),
            json.loads((base / "candidate_schema.json").read_text()))


@lru_cache(maxsize=1)
def sources():
    return json.loads((ROOT / "configs/sources.json").read_text())


def file_sha256(path):
    return _digest(str(Path(path).resolve()), Path(path).stat().st_size, Path(path).stat().st_mtime_ns)


@lru_cache(maxsize=256)
def _digest(path, size, mtime):
    h = sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1024*1024), b""):
            h.update(part)
    return h.hexdigest()


@dataclass(frozen=True)
class Sequence:
    sequence_id: int
    source_file: str
    key: str
    length: int
    format: str
    selector: dict = field(default_factory=dict)
    episode_id: int | None = None
    sequential: bool = False


def typed(value):
    array = np.asarray(value)
    result = {"value": array.tolist(), "dtype": str(array.dtype), "shape": list(array.shape)}
    return result


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)  # raw NaN/Inf explicitly represented, never zero-filled
    return value


class RawAdapter:
    dataset_id: str

    def sequences(self) -> list[Sequence]:
        raise NotImplementedError

    def sequence(self, index):
        if index < 0 or index >= len(self.sequences()):
            raise IndexError(f"sequence {index} outside [0, {len(self.sequences())})")
        return self.sequences()[index]

    def raw_fields(self, sequence, index):
        raise NotImplementedError

    def record(self, sequence_id, index):
        seq = self.sequence(sequence_id)
        if not 0 <= index < seq.length:
            raise IndexError(f"index {index} outside [0, {seq.length}) for {seq.key}")
        fields, extra = self.raw_fields(seq, index)
        return {"representation": "raw", "dataset_id": self.dataset_id,
                "sequence_id": seq.sequence_id, "sequence_key": seq.key,
                "episode_id": seq.episode_id, "source_file": seq.source_file,
                "source_file_sha256": file_sha256(ROOT / seq.source_file),
                "index": index, "fields": {k: typed(v) for k, v in fields.items()}, **extra}

    def global_row(self, row):
        if row < 0:
            raise IndexError("row must be >= 0")
        offset = 0
        for seq in self.sequences():
            if row < offset + seq.length:
                record = self.record(seq.sequence_id, row-offset)
                record["row_id"] = row
                return record
            offset += seq.length
        raise IndexError(f"row {row} outside [0, {offset})")

    def describe(self):
        raise NotImplementedError
