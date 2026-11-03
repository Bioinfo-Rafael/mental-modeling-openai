"""LLM bundle: ``weights/<key>.npy`` + ``weights/index.json`` (mounted read-only in the sandbox)
and ``manifest.json`` (harness side). Deterministic: the same checkpoint gives identical bytes."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mtd_rl_xavier.tools.output_safety import derived_path
from weightreading import SCHEMA_VERSION
from weightreading.tensors import Tensor

WEIGHTS_DIR = "weights"
INDEX_FILE = "index.json"
MANIFEST_FILE = "manifest.json"


class BundleError(ValueError):
    """An existing bundle differs from the one requested, or a bundle failed verification."""


@dataclass(frozen=True)
class Bundle:
    path: Path
    manifest_sha256: str

    @property
    def weights(self) -> Path:
        return self.path / WEIGHTS_DIR


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def npy_bytes(tensor: Tensor) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, tensor.to_numpy(), allow_pickle=False)
    return buffer.getvalue()


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _render(tensors: tuple[Tensor, ...], alias: str, checkpoint_sha256: str) -> dict[str, bytes]:
    """Relative path -> file bytes for the whole bundle."""
    files = {f"{WEIGHTS_DIR}/{t.name}.npy": npy_bytes(t) for t in tensors}
    files[f"{WEIGHTS_DIR}/{INDEX_FILE}"] = _json_bytes([t.name for t in tensors])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "network_alias": alias,
        "checkpoint_sha256": checkpoint_sha256,
        "files_sha256": {path: sha256_bytes(data) for path, data in sorted(files.items())},
        "tensors": [{"key": t.name, "dtype": t.dtype, "shape": list(t.shape), "raw_sha256": t.sha256} for t in tensors],
    }
    return {**files, MANIFEST_FILE: _json_bytes(manifest)}


def write_bundle(tensors: tuple[Tensor, ...], alias: str, checkpoint_sha256: str, root: Path) -> Bundle:
    """Write ``root/<alias>/`` atomically. An identical existing bundle is reused; a different one is refused."""
    target = derived_path(root / alias)
    files = _render(tensors, alias, checkpoint_sha256)
    manifest_sha = sha256_bytes(files[MANIFEST_FILE])
    if target.exists():
        verify_bundle(target, expected_manifest_sha256=manifest_sha)
        return Bundle(target, manifest_sha)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{alias}.partial-", dir=target.parent))
    try:
        (staging / WEIGHTS_DIR).mkdir()
        for relative, data in files.items():
            (staging / relative).write_bytes(data)
        os.rename(staging, target)
    except OSError:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return Bundle(target, manifest_sha)


def verify_bundle(path: Path, expected_manifest_sha256: str | None = None) -> Bundle:
    """Re-hash every bundle file against the manifest (e.g. after a sandbox session)."""
    manifest_bytes = (path / MANIFEST_FILE).read_bytes()
    manifest_sha = sha256_bytes(manifest_bytes)
    if expected_manifest_sha256 is not None and manifest_sha != expected_manifest_sha256:
        raise BundleError(f"existing bundle at {path} differs from the requested one; move it away first")
    recorded = json.loads(manifest_bytes)["files_sha256"]
    present = {p.relative_to(path).as_posix() for p in (path / WEIGHTS_DIR).rglob("*") if p.is_file()}
    if present != set(recorded):
        raise BundleError(f"bundle at {path} has unexpected or missing weight files")
    for relative, digest in recorded.items():
        if sha256_bytes((path / relative).read_bytes()) != digest:
            raise BundleError(f"bundle file changed: {relative}")
    return Bundle(path, manifest_sha)
