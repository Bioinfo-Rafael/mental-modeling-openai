"""Fixtures for the Translator tests: the committed 13_LAG episodes, loaded once per session."""

from __future__ import annotations

import pandas as pd
import pytest

from translate.lag.dataset import load_episode

from .tr_fakes import EPISODE_CSV, SHORT_LAST_STEP


@pytest.fixture(scope="session")
def frames() -> dict[str, pd.DataFrame]:
    """episode id -> hash-verified frame. Tests must not mutate these frames."""
    return {episode_id: load_episode(path) for episode_id, path in EPISODE_CSV.items()}


@pytest.fixture(scope="session")
def short_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The first episode cut at step SHORT_LAST_STEP: real rows, fast real rendering."""
    frame = frames["mhi_v1.3_ic100601_win"]
    return frame[frame["step"] <= SHORT_LAST_STEP]
