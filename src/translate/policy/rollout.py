"""The recorded rollout a policy input is read from (``<rollout_dir>/manifest.json`` + ``episode_seed<seed>.npz``).

Schema ``weightreading_rollout_v1``, protocol ``per_agent_v1``: one recurrent row per learner.
Only the policy arrays (``obs``, ``rnn_in``, ``logits``, ``gru_out``, ``actions``) and ``alive`` are
used. The manifest names the actor checkpoint, which is read in place and never copied.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROLLOUT_SCHEMA = "weightreading_rollout_v1"
ROLLOUT_PROTOCOL = "per_agent_v1"
MANIFEST = "manifest.json"
POLICY_ARRAYS = ("obs", "rnn_in", "logits", "gru_out", "actions", "alive")


class RolloutError(ValueError):
    """The rollout recording cannot be used as a policy input."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Rollout:
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

    def learner_row(self, aircraft: str) -> int:
        """The aircraft's recurrent row; only learners of the recorded policy have one."""
        if aircraft not in self.learner_uids:
            raise RolloutError(f"{aircraft} is not a learner of this policy")
        return self.learner_uids.index(aircraft)


def _checkpoint(manifest: dict) -> tuple[Path, str]:
    actors = manifest.get("actor_by_agent")
    if not isinstance(actors, dict) or not actors:
        raise RolloutError("rollout manifest has no actor_by_agent")
    pairs = {(entry["actor_file"], entry["sha256"]) for entry in actors.values()}
    if len(pairs) != 1:
        raise RolloutError("learners use different actor files; one shared actor is expected")
    path, digest = pairs.pop()
    return Path(path), digest


def load_rollout(rollout_dir: Path, seed: int) -> Rollout:
    manifest_path = rollout_dir / MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != ROLLOUT_SCHEMA or manifest.get("protocol") != ROLLOUT_PROTOCOL:
        raise RolloutError(f"rollout must be schema {ROLLOUT_SCHEMA}, protocol {ROLLOUT_PROTOCOL}")
    files = {entry["seed"]: entry["file"] for entry in manifest.get("episodes", [])}
    if seed not in files:
        raise RolloutError(f"seed {seed} is not in the rollout manifest")
    path = rollout_dir / files[seed]
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.array(data[key]) for key in POLICY_ARRAYS}
    checkpoint_path, checkpoint_sha = _checkpoint(manifest)
    return Rollout(
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
