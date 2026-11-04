"""Fixtures for the LLM-part tests (fakes live in ``llm_fakes.py``)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

SECURE_MODE = 0o700
SYSTEMONE_KEY_ENV = "TEST_SYSTEMONE_KEY"
SYSTEMONE_KEY_VALUE = "local-secret-key-0123456789"
OPENAI_KEY_VALUE = "sk-test-SECRET-value-0123"


@pytest.fixture
def secure_uds_path(tmp_path: Path) -> Path:
    """A socket path whose parent directory is owned by the user with mode 0700."""
    socket_dir = tmp_path / "vllm"
    socket_dir.mkdir()
    os.chmod(socket_dir, SECURE_MODE)
    return socket_dir / "clef.sock"


@pytest.fixture
def systemone_key(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv(SYSTEMONE_KEY_ENV, SYSTEMONE_KEY_VALUE)
    return SYSTEMONE_KEY_ENV
