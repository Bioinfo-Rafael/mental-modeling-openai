#!/usr/bin/env python3
"""Replay recorded MHI 2v2 rollouts through weightreading's numpy float64 reference forward.

numpy only (run with ``src`` on ``PYTHONPATH``); reads a directory written by
``weightreading_record_rollouts.py`` and writes ``replay_validation.json`` next to it. Two replays:

* teacher-forced: every decision starts from the recorded torch input state ``rnn_in[t]``, so the
  per-step numerical error is isolated;
* free-running: the reference carries its own state from ``h = 0`` at the start of the episode
  (what an activation extraction does), so any accumulated drift shows up.

``per_agent_v1`` steps both learner rows as one batch with one state row each; ``mhi_render_v1_3``
steps the two rows in packed order through a single shared state row (mtd-rl-lag ``GRULayer``
with ``N = 1, T = 2``). Exit 0 = recorded actions reproduced and diffs within ``TOL_FLOAT32``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from weightreading.reference import ReferencePolicy, build_policy
from weightreading.spec import check_against_tensors, load_spec
from weightreading.torch_reader import read_checkpoint

TOL_FLOAT32 = 1e-4  # float32 torch vs float64 numpy, as in weightreading_forward_parity.py
DEFAULT_SPEC = Path("configs/weightreading/lag_mappo_2v2_33.spec.json")
REPORT_NAME = "replay_validation.json"


def replay_decision(policy: ReferencePolicy, obs: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, ...]:
    """One team decision. ``obs`` (rows, inputs); ``state`` (R, hidden) with R = rows or R = 1.

    Returns ``(logits (rows, bins), per-row hidden (rows, hidden), carried state (R, hidden))``.
    """
    if state.shape[0] == obs.shape[0]:
        out = policy.step(obs, state)
        return np.concatenate(out.logits, axis=-1), out.hidden, out.hidden
    logits, hidden = [], []
    carried = state
    for row in obs:
        out = policy.step(row[None], carried)
        logits.append(np.concatenate(out.logits, axis=-1)[0])
        hidden.append(out.hidden[0])
        carried = out.hidden
    return np.stack(logits), np.stack(hidden), carried


def head_argmax(logits: np.ndarray, head_sizes: list[int]) -> np.ndarray:
    """(..., sum(bins)) logits -> (..., heads) argmax per head."""
    splits = np.split(logits, np.cumsum(head_sizes)[:-1], axis=-1)
    return np.stack([part.argmax(axis=-1) for part in splits], axis=-1)


def min_top2_margin(logits: np.ndarray, head_sizes: list[int]) -> float:
    """Smallest gap between the best and second-best logit of any head (near-tie indicator)."""
    splits = np.split(logits, np.cumsum(head_sizes)[:-1], axis=-1)
    margins = [np.diff(np.sort(part, axis=-1)[..., -2:], axis=-1).min() for part in splits]
    return float(min(margins))


def replay_episode(policy: ReferencePolicy, data: Any, head_sizes: list[int], free_running: bool) -> dict:
    obs, rnn_in = data["obs"].astype(np.float64), data["rnn_in"][:, :, 0].astype(np.float64)
    steps = obs.shape[0]
    logits = np.zeros(data["logits"].shape)
    hidden = np.zeros(data["gru_out"].shape)
    carried = np.zeros(rnn_in.shape)
    state = np.zeros_like(rnn_in[0])
    for t in range(steps):
        start = state if free_running else rnn_in[t]
        logits[t], hidden[t], carried[t] = replay_decision(policy, obs[t], start)
        state = carried[t]
    actions = data["actions"]
    agree = head_argmax(logits, head_sizes) == actions
    return {
        "action_agreement": float(agree.mean()),
        "action_agreement_per_head": [float(v) for v in agree.mean(axis=(0, 1))],
        "action_mismatches": int((~agree).sum()),
        "logits_max_abs_diff": float(np.abs(logits - data["logits"]).max()),
        "row_hidden_max_abs_diff": float(np.abs(hidden - data["gru_out"]).max()),
        "carried_state_max_abs_diff": float(np.abs(carried - data["rnn_out"][:, :, 0]).max()),
        "reference_min_top2_logit_margin": min_top2_margin(logits, head_sizes),
    }


def recording_checks(data: Any, head_sizes: list[int]) -> dict:
    """Internal consistency of the recording itself (torch side only)."""
    rnn_in, rnn_out = data["rnn_in"], data["rnn_out"]
    return {
        "decisions": int(data["obs"].shape[0]),
        "initial_state_zero": bool(np.all(rnn_in[0] == 0)),
        "state_continuity_max_abs_diff": float(np.abs(rnn_in[1:] - rnn_out[:-1]).max()) if len(rnn_in) > 1 else 0.0,
        "final_state_equals_last_rnn_out": bool(np.array_equal(data["final_rnn_state"], rnn_out[-1])),
        "masks_all_ones": bool(np.all(data["masks"] == 1)),
        "torch_argmax_equals_actions": float((head_argmax(data["logits"], head_sizes) == data["actions"]).mean()),
        "torch_min_top2_logit_margin": min_top2_margin(data["logits"].astype(np.float64), head_sizes),
    }


def episode_ok(record: dict, teacher: dict, free: dict) -> bool:
    within = [teacher[k] < TOL_FLOAT32 for k in ("logits_max_abs_diff", "row_hidden_max_abs_diff")]
    within.append(teacher["carried_state_max_abs_diff"] < TOL_FLOAT32)
    exact = [teacher["action_agreement"] == 1.0, free["action_agreement"] == 1.0]
    exact += [record["initial_state_zero"], record["state_continuity_max_abs_diff"] == 0.0]
    exact += [record["final_state_equals_last_rnn_out"], record["masks_all_ones"]]
    exact.append(record["torch_argmax_equals_actions"] == 1.0)
    return all(within + exact)


def load_reference(manifest: dict, spec_path: Path) -> tuple[ReferencePolicy, str]:
    checkpoint_path = Path(manifest["provenance"]["checkpoint"])
    checkpoint = read_checkpoint(checkpoint_path)
    if checkpoint.sha256 != manifest["provenance"]["checkpoint_sha256"]:
        raise ValueError("checkpoint on disk differs from the one recorded in manifest.json")
    spec = load_spec(spec_path)
    check_against_tensors(spec, checkpoint.tensors, checkpoint.sha256)
    return build_policy(spec, checkpoint.tensors), checkpoint.sha256


def crosscheck_lag_evaluation(rollout_dir: Path, manifest: dict, eval_root: Path) -> dict:
    """Compare against mtd-rl-lag ``evaluate_mhi_team.py`` run separately (``<eval_root>/<seed>/``).

    Bitwise-equal final recurrent state, equal step count and equal match metrics show that the
    recorder's hooks and its in-process episode loop did not change the rollout.
    """
    results = {}
    for episode in manifest["episodes"]:
        seed_dir = eval_root / str(episode["seed"])
        evaluation = json.loads((seed_dir / "evaluation.json").read_text(encoding="utf-8"))
        with np.load(seed_dir / "final_state.npz") as final, np.load(rollout_dir / episode["file"]) as data:
            rnn_equal = bool(np.array_equal(final["rnn"], data["final_rnn_state"]))
        metrics = {k: v for k, v in evaluation["metrics"].items() if k != "launches"}
        recorded = {k: v for k, v in episode["metrics"].items() if k != "launches"}
        results[str(episode["seed"])] = {
            "final_rnn_state_bitwise_equal": rnn_equal,
            "steps_equal": evaluation["steps"] == episode["final_env_step"],
            "metrics_equal": json.dumps(metrics, sort_keys=True) == json.dumps(recorded, sort_keys=True),
            "launch_count_equal": len(evaluation["metrics"]["launches"]) == len(episode["metrics"]["launches"]),
        }
    return {"eval_root": str(eval_root), "seeds": results, "ok": all(all(v.values()) for v in results.values())}


def run(rollout_dir: Path, spec_path: Path, lag_eval_root: Path | None = None) -> dict:
    manifest = json.loads((rollout_dir / "manifest.json").read_text(encoding="utf-8"))
    policy, digest = load_reference(manifest, spec_path)
    head_sizes = list(manifest["action_head_sizes"])
    episodes = []
    for episode in manifest["episodes"]:
        with np.load(rollout_dir / episode["file"]) as data:
            record = recording_checks(data, head_sizes)
            teacher = replay_episode(policy, data, head_sizes, free_running=False)
            free = replay_episode(policy, data, head_sizes, free_running=True)
        ok = episode_ok(record, teacher, free)
        episodes.append(
            {
                "file": episode["file"],
                "seed": episode["seed"],
                "recording": record,
                "teacher_forced": teacher,
                "free_running": free,
                "ok": ok,
            }
        )
    worst = {
        mode: {
            key: max(e[mode][key] for e in episodes)
            for key in ("logits_max_abs_diff", "row_hidden_max_abs_diff", "carried_state_max_abs_diff")
        }
        | {"action_agreement_min": min(e[mode]["action_agreement"] for e in episodes)}
        for mode in ("teacher_forced", "free_running")
    }
    report: dict[str, Any] = {
        "rollout_dir": str(rollout_dir),
        "protocol": manifest["protocol"],
        "spec": str(spec_path),
        "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        "checkpoint_sha256": digest,
        "reference": "weightreading.reference.ReferencePolicy (numpy float64)",
        "tolerance": TOL_FLOAT32,
        "summary": worst,
        "episodes": episodes,
        "ok": all(e["ok"] for e in episodes),
    }
    if lag_eval_root is None:
        return report
    crosscheck = crosscheck_lag_evaluation(rollout_dir, manifest, lag_eval_root)
    return {**report, "lag_evaluate_mhi_team_crosscheck": crosscheck, "ok": report["ok"] and crosscheck["ok"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("rollout_dir", type=Path)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--lag-eval-root", type=Path, help="evaluate_mhi_team.py outputs, one <seed>/ per episode")
    args = parser.parse_args()
    report = run(args.rollout_dir.resolve(), args.spec, args.lag_eval_root)
    out = args.rollout_dir / REPORT_NAME
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "ok": report["ok"], "summary": report["summary"]}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
