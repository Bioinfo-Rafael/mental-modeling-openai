"""Every judge test scans and marks tmp_path/evaluation, never the repository's outputs/evaluation."""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation import cli


@pytest.fixture(autouse=True)
def guard_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "evaluation"
    monkeypatch.setattr(cli, "GUARD_ROOT", root)
    return root
