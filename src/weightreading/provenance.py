"""Provenance records: SHA-256 of inputs and of this package's code, plus environment facts."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path

import weightreading

GIT_TIMEOUT_S = 10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def code_hashes() -> dict[str, str]:
    """``weightreading/<file>.py`` -> sha256 for every module of this package."""
    package = Path(weightreading.__file__).resolve().parent
    return {f"weightreading/{p.name}": sha256_file(p) for p in sorted(package.glob("*.py"))}


def git_state(root: Path) -> dict[str, object]:
    """Commit and dirty flag of the repository, or nulls when git is unavailable."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=GIT_TIMEOUT_S
        )
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--", "src/weightreading", "configs/weightreading"],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"commit": None, "weightreading_dirty": None}
    if commit.returncode != 0:
        return {"commit": None, "weightreading_dirty": None}
    return {"commit": commit.stdout.strip(), "weightreading_dirty": bool(status.stdout.strip())}


def environment() -> dict[str, str]:
    return {
        "harness_python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
    }


def input_record(checkpoint: Path, checkpoint_sha256: str, spec_path: Path, manifest_sha256: str) -> dict[str, str]:
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "spec_path": str(spec_path),
        "spec_sha256": sha256_file(spec_path),
        "bundle_manifest_sha256": manifest_sha256,
    }
