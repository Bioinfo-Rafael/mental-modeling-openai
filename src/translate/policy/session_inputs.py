"""Read-only inputs of one intent-B session, laid out like a bundle so ``verify_bundle`` re-hashes them.

``<dir>/weights/<W key>.npy`` + ``weights/index.json`` (the checkpoint tensors, as in the bundle),
``<dir>/weights/activations/<A key>.npy`` + ``activations/index.json`` (one aircraft's window) and
``<dir>/manifest.json`` (harness side, not mounted). The sandbox mounts ``<dir>/weights`` read-only
at ``/work/weights`` and preloads ``A`` next to ``W``.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from translate.output_safety import derived_path
from translate.policy import SCHEMA_VERSION
from translate.policy.bundle import INDEX_FILE, MANIFEST_FILE, WEIGHTS_DIR, Bundle, npy_bytes, sha256_bytes
from translate.policy.tensors import Tensor

ACTIVATIONS_DIR = "activations"
ACTIVATIONS_NAME = "A"
PRELOAD = ((ACTIVATIONS_NAME, ACTIVATIONS_DIR),)


@dataclass(frozen=True)
class SessionInputs:
    bundle: Bundle
    files_sha256: dict[str, str]


def _array_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def render_inputs(tensors: tuple[Tensor, ...], activations: dict[str, np.ndarray], metadata: dict) -> dict[str, bytes]:
    """Relative path -> bytes. Deterministic for identical inputs."""
    files = {f"{WEIGHTS_DIR}/{t.name}.npy": npy_bytes(t) for t in tensors}
    files[f"{WEIGHTS_DIR}/{INDEX_FILE}"] = _json_bytes([t.name for t in tensors])
    prefix = f"{WEIGHTS_DIR}/{ACTIVATIONS_DIR}"
    files.update({f"{prefix}/{key}.npy": _array_bytes(array) for key, array in activations.items()})
    files[f"{prefix}/{INDEX_FILE}"] = _json_bytes(list(activations))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "intent-b-session-inputs",
        **metadata,
        "activations": {key: {"shape": list(a.shape), "dtype": str(a.dtype)} for key, a in activations.items()},
        "files_sha256": {path: sha256_bytes(data) for path, data in sorted(files.items())},
    }
    return {**files, MANIFEST_FILE: _json_bytes(manifest)}


def write_inputs(
    directory: Path, tensors: tuple[Tensor, ...], activations: dict[str, np.ndarray], metadata: dict
) -> SessionInputs:
    """Write ``directory`` atomically; it must not exist yet."""
    target = derived_path(directory)
    if target.exists():
        raise FileExistsError(f"{target} already exists")
    files = render_inputs(tensors, activations, metadata)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.partial-", dir=target.parent))
    try:
        (staging / WEIGHTS_DIR / ACTIVATIONS_DIR).mkdir(parents=True)
        for relative, data in files.items():
            (staging / relative).write_bytes(data)
        os.rename(staging, target)
    except OSError:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    manifest = files[MANIFEST_FILE]
    recorded = json.loads(manifest)["files_sha256"]
    return SessionInputs(Bundle(target, sha256_bytes(manifest)), recorded)
