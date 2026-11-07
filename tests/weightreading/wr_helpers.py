"""Test helpers: torch-format checkpoints emitted from pickle opcodes (no torch), a small spec."""

from __future__ import annotations

import io
import os
import pickle
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from weightreading.architecture import expected_params
from weightreading.spec import parse_spec

LAG_ROOT = Path(__file__).resolve().parents[3] / "mtd-rl-lag"  # the sibling mtd-rl-lag checkout
LAG_MODEL_DIR = Path(os.environ.get("WEIGHTREADING_LAG_MODEL_DIR", str(LAG_ROOT / "envs" / "JSBSim" / "model")))
# Delivered 2v2 MAPPO actors (client files; read-only, never copied into this repository).
MHI_MODEL_DIR = Path(os.environ.get("WEIGHTREADING_MHI_MODEL_DIR", str(LAG_ROOT / "models")))
MHI_ACTORS = ("v1.3/files/actor_latest.pt", "v1.3-1/files/actor_latest.pt")
PREFIX = "archive"


# --- pickle opcode emitters -------------------------------------------------------------------


def op_str(text: str) -> bytes:
    data = text.encode("utf-8")
    return pickle.BINUNICODE + struct.pack("<I", len(data)) + data


def op_int(value: int) -> bytes:
    return pickle.BININT + struct.pack("<i", value)


def op_global(module: str, name: str) -> bytes:
    return pickle.GLOBAL + f"{module}\n{name}\n".encode()


def op_tuple(items: list[bytes]) -> bytes:
    return pickle.MARK + b"".join(items) + pickle.TUPLE


def op_empty_ordered_dict() -> bytes:
    return op_global("collections", "OrderedDict") + pickle.EMPTY_TUPLE + pickle.REDUCE


@dataclass(frozen=True)
class Entry:
    name: str
    storage_key: str
    storage_numel: int
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_type: str = "FloatStorage"


def contiguous(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides, step = [], 1
    for size in reversed(shape):
        strides.append(step)
        step *= size
    return tuple(reversed(strides))


def op_tensor(entry: Entry) -> bytes:
    pid = op_tuple(
        [
            op_str("storage"),
            op_global("torch", entry.storage_type),
            op_str(entry.storage_key),
            op_str("cpu"),
            op_int(entry.storage_numel),
        ]
    )
    args = op_tuple(
        [
            pid + pickle.BINPERSID,
            op_int(entry.offset),
            op_tuple([op_int(v) for v in entry.shape]),
            op_tuple([op_int(v) for v in entry.stride]),
            pickle.NEWFALSE,
            op_empty_ordered_dict(),
        ]
    )
    return op_global("torch._utils", "_rebuild_tensor_v2") + args + pickle.REDUCE


def state_dict_pickle(items: list[tuple[bytes, bytes]], tail: bytes = b"") -> bytes:
    """``items`` are (key opcodes, value opcodes); ``tail`` goes before STOP (e.g. a BUILD)."""
    body = b"".join(key + value for key, value in items)
    return pickle.PROTO + b"\x02" + op_empty_ordered_dict() + pickle.MARK + body + pickle.SETITEMS + tail + pickle.STOP


def build_zip(members: dict[str, bytes], compression: int = zipfile.ZIP_STORED) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=compression) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def checkpoint_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """A torch-format checkpoint with one float32 storage per array."""
    entries, members = [], {}
    for position, (name, array) in enumerate(arrays.items()):
        values = np.ascontiguousarray(array, dtype="<f4")
        key = str(position)
        entries.append(Entry(name, key, values.size, 0, values.shape, contiguous(values.shape)))
        members[f"{PREFIX}/data/{key}"] = values.tobytes()
    payload = state_dict_pickle([(op_str(e.name), op_tensor(e)) for e in entries])
    return build_zip({f"{PREFIX}/data.pkl": payload, f"{PREFIX}/version": b"3\n", **members})


# --- small spec + matching weights --------------------------------------------------------------

SMALL_SPEC = {
    "schema_version": 1,
    "network_alias": "net-0123abcd",
    "hidden_from_prompt": ["provenance", "notes"],
    "provenance": {"origin": "mtd-rl-lag envs/JSBSim/model/baseline_model.pt"},
    "description": "Low-level aircraft flight controller test network.",
    "inputs": [
        {"index": 0, "name": "roll_sin", "definition": "sin(roll)", "unit": ""},
        {"index": 1, "name": "pitch_sin", "definition": "sin(pitch)", "unit": ""},
        {"index": 2, "name": "airspeed", "definition": "airspeed / 340", "unit": "340 m/s"},
    ],
    "input_clip": [-10.0, 10.0],
    "architecture": [
        {"id": "base", "type": "mlp", "prefix": "base.mlp.fc.", "input": "observation", "input_size": 3,
         "hidden_sizes": [4, 4]},
        {"id": "gru", "type": "gru", "prefix": "rnn.gru.", "input": "base", "input_size": 4, "hidden_size": 4},
        {"id": "gru_norm", "type": "layernorm", "prefix": "rnn.norm.", "input": "gru", "size": 4},
        {"id": "heads", "type": "categorical_heads", "prefix": "act.action_outs.", "input": "gru_norm",
         "input_size": 4},
    ],
    "outputs": [
        {"head": 0, "name": "aileron", "bins": 3, "bin_value": {"linspace": [-1.0, 1.0]}},
        {"head": 1, "name": "throttle", "bins": 2, "bin_value": {"linspace": [0.4, 0.9]}},
    ],
    "notes": ["secret note: trained by baseline"],
}  # fmt: skip


def small_arrays(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    spec = parse_spec(SMALL_SPEC)
    shapes = {key: shape for block in spec.blocks for key, shape in expected_params(block).items()}
    return {key: rng.standard_normal(shape).astype(np.float32) for key, shape in shapes.items()}
