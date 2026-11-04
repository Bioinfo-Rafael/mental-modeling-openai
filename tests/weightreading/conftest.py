"""Fixtures for weightreading tests (helpers live in ``wr_helpers.py``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .wr_helpers import LAG_MODEL_DIR, MHI_ACTORS, MHI_MODEL_DIR, SMALL_SPEC, checkpoint_bytes, small_arrays


@pytest.fixture
def small_spec_dict() -> dict:
    return json.loads(json.dumps(SMALL_SPEC))


@pytest.fixture
def small_files(tmp_path: Path) -> tuple[Path, Path]:
    """(checkpoint path, spec path) for the small network."""
    checkpoint = tmp_path / "small.pt"
    checkpoint.write_bytes(checkpoint_bytes(small_arrays()))
    spec = tmp_path / "small.spec.json"
    spec.write_text(json.dumps(SMALL_SPEC), encoding="utf-8")
    return checkpoint, spec


@pytest.fixture
def lag_model_dir() -> Path:
    if not (LAG_MODEL_DIR / "baseline_model.pt").exists():
        pytest.skip("mtd-rl-lag checkpoints not available (set WEIGHTREADING_LAG_MODEL_DIR)")
    return LAG_MODEL_DIR


@pytest.fixture
def mhi_model_dir() -> Path:
    if not all((MHI_MODEL_DIR / actor).exists() for actor in MHI_ACTORS):
        pytest.skip("delivered 2v2 actors not available (set WEIGHTREADING_MHI_MODEL_DIR)")
    return MHI_MODEL_DIR
