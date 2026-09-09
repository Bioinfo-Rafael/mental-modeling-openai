from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def episode_path(tmp_path: Path) -> Path:
    path = tmp_path / "episode.npz"
    np.savez(
        path,
        states=np.asarray(
            [[[-0.5, 0.0]], [[-0.49, 0.01]], [[-0.47, 0.02]], [[-0.46, 0.01]],
             [[-0.44, 0.02]], [[-0.41, 0.03]], [[-0.37, 0.04]], [[-0.32, 0.05]]],
            dtype=np.float32,
        ),
        actions=np.asarray([[0], [1], [2], [1], [2], [2], [1], [2]], dtype=np.int64),
        rewards=np.asarray([[-1.0]] * 8, dtype=np.float64),
        episodic_return=np.asarray([[-8.0]], dtype=np.float32),
    )
    return path

