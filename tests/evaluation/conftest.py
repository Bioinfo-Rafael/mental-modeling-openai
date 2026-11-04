"""Fixtures for the judge tests: the real 13_LAG episode and one fake-layer Translator run."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mtd_rl_xavier.lag.dataset import load_episode
from tests.translator.tr_fakes import EPISODE_CSV, WIN01

from .eval_fakes import write_translator_run


@pytest.fixture(scope="session")
def episode_frame() -> pd.DataFrame:
    """The full, hash-verified example episode. Tests must not mutate it."""
    return load_episode(EPISODE_CSV[WIN01])


@pytest.fixture(scope="session")
def translator_run(tmp_path_factory: pytest.TempPathFactory, episode_frame: pd.DataFrame) -> Path:
    """A Translator run of the example scenario: real episode and renderers, fake layers.

    Five translation events (``eval_fakes.EVENT_IDS``). Read-only: tests that tamper copy it first.
    """
    run_dir = tmp_path_factory.mktemp("translator") / "run"
    summary = write_translator_run(run_dir, episode_frame)
    assert summary.status == "completed"
    return run_dir
