"""The recorded episode of the comparison and the (step, aircraft) session plan of method B.

Reads ``<rollout_dir>/manifest.json`` and ``episode_seed<seed>.npz`` (schema
``weightreading_rollout_v1``, protocol ``per_agent_v1``: one recurrent row per learner). Only the
policy arrays (``obs``, ``rnn_in``, ``logits``, ``gru_out``, ``actions``) and ``alive`` (to skip dead targets,
never shown to the model) are used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from weightreading.intent_config import CompareConfig
from weightreading.intent_trace import window_bounds
from weightreading.provenance import sha256_file

ROLLOUT_SCHEMA = "weightreading_rollout_v1"
ROLLOUT_PROTOCOL = "per_agent_v1"
MANIFEST = "manifest.json"
POLICY_ARRAYS = ("obs", "rnn_in", "logits", "gru_out", "actions", "alive")


class EpisodeError(ValueError):
    """The rollout recording cannot be used for method B."""


@dataclass(frozen=True)
class Episode:
    path: Path
    sha256: str
    manifest_path: Path
    manifest_sha256: str
    aircraft_uids: tuple[str, ...]
    learner_uids: tuple[str, ...]
    checkpoint_path: Path
    checkpoint_sha256: str
    arrays: dict[str, np.ndarray]

    @property
    def decisions(self) -> int:
        return int(self.arrays["obs"].shape[0])


@dataclass(frozen=True)
class PlannedSession:
    step: int
    aircraft: str
    row: int  # learner row in the recording (-1 when skipped before the row is known)
    window: tuple[int, int]
    skipped: str | None  # reason, or None when the session runs

    @property
    def name(self) -> str:
        return f"s{self.step:04d}_{self.aircraft}"


def _checkpoint(manifest: dict) -> tuple[Path, str]:
    actors = manifest.get("actor_by_agent")
    if not isinstance(actors, dict) or not actors:
        raise EpisodeError("rollout manifest has no actor_by_agent")
    pairs = {(entry["actor_file"], entry["sha256"]) for entry in actors.values()}
    if len(pairs) != 1:
        raise EpisodeError("learners use different actor files; method B expects one shared actor")
    path, digest = pairs.pop()
    return Path(path), digest


def load_episode(compare: CompareConfig) -> Episode:
    manifest_path = compare.rollout_dir / MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != ROLLOUT_SCHEMA or manifest.get("protocol") != ROLLOUT_PROTOCOL:
        raise EpisodeError(f"rollout must be schema {ROLLOUT_SCHEMA}, protocol {ROLLOUT_PROTOCOL}")
    files = {entry["seed"]: entry["file"] for entry in manifest.get("episodes", [])}
    if compare.seed not in files:
        raise EpisodeError(f"seed {compare.seed} is not in the rollout manifest")
    path = compare.rollout_dir / files[compare.seed]
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.array(data[key]) for key in POLICY_ARRAYS}
    checkpoint_path, checkpoint_sha = _checkpoint(manifest)
    return Episode(
        path=path,
        sha256=sha256_file(path),
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        aircraft_uids=tuple(manifest["aircraft_uids"]),
        learner_uids=tuple(manifest["learner_uids"]),
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        arrays=arrays,
    )


def plan_session(episode: Episode, step: int, aircraft: str, window_steps: int) -> PlannedSession:
    window = window_bounds(step, window_steps)
    if aircraft not in episode.learner_uids:
        return PlannedSession(step, aircraft, -1, window, f"{aircraft} is not a learner of this policy")
    row = episode.learner_uids.index(aircraft)
    if step >= episode.decisions:
        return PlannedSession(step, aircraft, row, window, f"step {step} is after the episode end")
    if not bool(episode.arrays["alive"][step, episode.aircraft_uids.index(aircraft)]):
        return PlannedSession(step, aircraft, row, window, f"{aircraft} is not alive at step {step}")
    return PlannedSession(step, aircraft, row, window, None)


def plan_sessions(episode: Episode, compare: CompareConfig) -> tuple[PlannedSession, ...]:
    """By step, then aircraft in ``targets`` order, so each aircraft's previous translation exists."""
    return tuple(
        plan_session(episode, step, aircraft, compare.window_steps)
        for step in compare.steps
        for aircraft in compare.targets
    )
