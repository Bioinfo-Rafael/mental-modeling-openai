"""Synthetic method-B inputs: a recorded episode of the small network, its manifest and a compare.yaml."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from translate import paths
from weightreading.intent_trace import concatenated_logits, trace_episode
from weightreading.provenance import sha256_file
from weightreading.reference import build_policy
from weightreading.spec import load_spec
from weightreading.torch_reader import read_checkpoint

SEED = 7
DECISIONS = 8
STEPS = [3, 6]
WINDOW = 4
DEAD_FROM = 5  # A0200 is not alive from this step on
AIRCRAFT = ["A0100", "A0200", "B0100", "B0200"]
LEARNERS = ["A0100", "A0200"]
MODEL = "fake-model"
EFFORT = "high"
INSTRUCTION = "No operator instruction was given."
RUN_CONFIG = paths.CONFIGS / "weightreading" / "intent_b.yaml"


@dataclass(frozen=True)
class IntentFiles:
    compare: Path
    config: Path
    spec: Path
    checkpoint: Path
    cues_root: Path
    output_root: Path


def write_rollout(directory: Path, checkpoint: Path, spec_path: Path) -> None:
    """Record the small network on random observations, as the real recorder would."""
    policy = build_policy(load_spec(spec_path), read_checkpoint(checkpoint).tensors)
    obs = np.random.default_rng(SEED).standard_normal((DECISIONS, len(LEARNERS), policy.input_size)).astype(np.float32)
    trace = trace_episode(policy, obs)
    alive = np.ones((DECISIONS, len(AIRCRAFT)), dtype=bool)
    alive[DEAD_FROM:, 1] = False
    directory.mkdir(parents=True)
    np.savez(
        directory / f"episode_seed{SEED}.npz",
        obs=obs,
        rnn_in=trace["h_prev"][:, :, None, :].astype(np.float32),
        logits=concatenated_logits(trace, policy).astype(np.float32),
        gru_out=trace["gru.h_new"].astype(np.float32),
        actions=trace["action"].astype(np.int64),
        alive=alive,
    )
    actor = {"actor_file": str(checkpoint), "sha256": sha256_file(checkpoint)}
    manifest = {
        "schema": "weightreading_rollout_v1",
        "protocol": "per_agent_v1",
        "aircraft_uids": AIRCRAFT,
        "learner_uids": LEARNERS,
        "actor_by_agent": {uid: actor for uid in LEARNERS},
        "episodes": [{"seed": SEED, "file": f"episode_seed{SEED}.npz"}],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def write_intent_files(tmp_path: Path, checkpoint: Path, spec_path: Path) -> IntentFiles:
    rollout = tmp_path / "rollout"
    write_rollout(rollout, checkpoint, spec_path)
    compare = {
        "seed": SEED,
        "rollout_dir": str(rollout),
        "steps": STEPS,
        "targets": LEARNERS,
        "window_steps": WINDOW,
        "step_seconds": 0.2,
        "model": MODEL,
        "reasoning_effort": EFFORT,
        "instruction": INSTRUCTION,
        "clip_half_width_s": 30,
        "output_root": str(tmp_path / "intent_compare"),
    }
    compare_path = tmp_path / "compare.yaml"
    compare_path.write_text(yaml.safe_dump(compare), encoding="utf-8")
    config = yaml.safe_load(RUN_CONFIG.read_text(encoding="utf-8"))
    config = {**config, "model": MODEL, "pricing": {MODEL: {"input": 2.0, "cached_input": 0.1, "output": 10.0}}}
    config_path = tmp_path / "intent_b.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return IntentFiles(compare_path, config_path, spec_path, checkpoint, tmp_path / "intent_compare", tmp_path / "wr")
