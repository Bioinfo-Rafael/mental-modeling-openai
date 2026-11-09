"""Synthetic policy-input material: a recorded episode of the small network and a run config around it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from translate import paths
from translate.config import RunConfig, parse_config
from translate.events import TARGET_AIRCRAFT, EpisodeRef, Instruction, Scenario, TargetRef
from translate.inputs import RunInputs
from translate.lag.rendering import render_lag17_jsonl
from translate.policy.material import PolicyMaterial
from translate.policy.reference import build_policy
from translate.policy.rollout import sha256_file
from translate.policy.spec import load_spec
from translate.policy.torch_reader import read_checkpoint
from translate.policy.trace import concatenated_logits, trace_episode
from translate.scenario import EpisodeFacts

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
POLICY_CONFIG = paths.CONFIGS / "translate" / "seed100601-policy.yaml"


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


def policy_raw(rollout: Path, spec: Path, config: Path = POLICY_CONFIG, **sections: Any) -> dict[str, Any]:
    """A shipped policy config pointed at the synthetic rollout, spec and model; ``sections`` replace whole keys."""
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    policy = {"rollout_dir": str(rollout), "seed": SEED, "spec": str(spec), "checkpoint": None, "past_steps": WINDOW}
    translation = {**raw["translation"], "model": MODEL, "reasoning_effort": EFFORT}
    pricing = {MODEL: {"input": 2.0, "cached_input": 0.1, "output": 10.0}}
    return {**raw, "policy": policy, "translation": translation, "pricing": pricing, **sections}


def policy_config(rollout: Path, spec: Path, config: Path = POLICY_CONFIG, **sections: Any) -> RunConfig:
    return parse_config(policy_raw(rollout, spec, config, **sections), path=config, sha256="test")


def synthetic_scenario() -> Scenario:
    instructions = tuple(
        Instruction(f"neutral-{name}", 0, TargetRef(TARGET_AIRCRAFT, name), INSTRUCTION) for name in LEARNERS
    )
    return Scenario(
        2, "synthetic", EpisodeRef("synthetic", "episode_01", Path("unused.csv"), None), instructions, (), ()
    )


def policy_inputs(config: RunConfig, material: PolicyMaterial) -> RunInputs:
    """Run inputs of the synthetic episode (the policy prompt reads no trajectory)."""
    facts = EpisodeFacts("synthetic", DECISIONS - 1, {"A": tuple(LEARNERS)}, dict.fromkeys(LEARNERS))
    return RunInputs(pd.DataFrame(), synthetic_scenario(), facts, config, render_lag17_jsonl, material)
