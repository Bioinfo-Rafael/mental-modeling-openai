#!/usr/bin/env python3
"""Record per-decision policy inputs and outputs of an MHI 2v2 team actor in mtd-rl-lag.

Run with mtd-rl-lag's interpreter (``$LAG_ROOT/.venv/bin/python``; torch + JSBSim), never on the
login node: see ``slurm/lag/record_rollouts.sbatch``. torch is NOT a dependency of
this repository and nothing in mtd-rl-lag is modified.

Each episode mirrors ``scripts/eval/evaluate_mhi_team.py`` of mtd-rl-lag exactly: a fresh
``MultipleCombatEnv``, ``random``/``numpy``/``torch`` seeded with the episode seed, the pinned
actor loaded through ``load_team_bundle`` (policy manifest hash + world checks), and the real
``TeamSide`` / ``play_team_match`` loop. Capture is read-only: torch forward hooks on the actor
(inputs, actions, log-probs, output state), on its ``nn.GRU`` (per-row pre-LayerNorm ``h'``) and
on each head's ``logits_net`` (raw logits), plus a ``before_step`` / ``after_step`` callback that
reads the simulator state. No extra RNG draws happen.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rollout_schema import (
    CONTEXT_FIELDS,
    DECISION_FIELDS,
    DEFAULT_SCENARIO,
    DEFAULT_SEEDS,
    PROTOCOLS,
    SCHEMA,
    episode_file,
)

DEPENDENCIES = ("numpy", "torch", "gymnasium", "pymap3d", "jsbsim")
FINAL_STATE_FIELD = "(R, 1, 128) float32: TeamSide.rnn_state after the last decision (== rnn_out[-1])"


class ActorTap:
    """Forward hooks that capture one team decision of a ``PPOActor`` without changing it."""

    def __init__(self, actor: torch.nn.Module) -> None:
        self._current: dict[str, Any] = {}
        self._handles = [
            actor.register_forward_pre_hook(self._before_actor, with_kwargs=True),
            actor.register_forward_hook(self._after_actor, with_kwargs=True),
            actor.rnn.gru.register_forward_hook(self._after_gru),
        ]
        self._handles += [
            head.logits_net.register_forward_hook(self._head_hook(index))
            for index, head in enumerate(actor.act.action_outs)
        ]
        self.head_count = len(actor.act.action_outs)
        # act_in: the input every action head reads (the act MLP output); one pre-hook on the first head.
        self._handles.append(actor.act.action_outs[0].logits_net.register_forward_pre_hook(self._before_heads))

    def _before_heads(self, _module: torch.nn.Module, args: tuple) -> None:
        self._current["act_in"] = args[0].detach().cpu().numpy().astype(np.float32).copy()

    def _before_actor(self, _module: torch.nn.Module, args: tuple, kwargs: dict) -> None:
        obs, rnn_states, masks = args[:3]
        self._current = {
            "obs": np.array(obs, dtype=np.float32, copy=True),
            "rnn_in": np.array(rnn_states, dtype=np.float32, copy=True),
            "masks": np.array(masks, dtype=np.float32, copy=True),
            "deterministic": bool(kwargs.get("deterministic", False)),
            "action_mask_used": kwargs.get("action_mask") is not None,
            "gru_out": [],
            "logits": {},
        }

    def _after_gru(self, _module: torch.nn.Module, _args: tuple, output: tuple) -> None:
        sequence = output[0].detach().cpu().numpy()  # (T, N, hidden); rows in packed order
        self._current["gru_out"].append(sequence.reshape(-1, sequence.shape[-1]).copy())

    def _head_hook(self, index: int) -> Any:
        def hook(_module: torch.nn.Module, _args: tuple, output: torch.Tensor) -> None:
            self._current["logits"][index] = output.detach().cpu().numpy().copy()

        return hook

    def _after_actor(self, _module: torch.nn.Module, _args: tuple, _kwargs: dict, output: tuple) -> None:
        actions, log_probs, rnn_out = output
        self._current["actions"] = actions.detach().cpu().numpy().astype(np.int64)
        self._current["log_probs"] = log_probs.detach().cpu().numpy().astype(np.float32)
        self._current["rnn_out"] = rnn_out.detach().cpu().numpy().astype(np.float32)

    def take(self) -> dict[str, np.ndarray]:
        """The decision captured since the last call, as ``DECISION_FIELDS`` arrays."""
        current, self._current = self._current, {}
        if "actions" not in current or "act_in" not in current or len(current["logits"]) != self.head_count:
            raise RuntimeError("actor forward was not fully captured for this decision")
        if current["action_mask_used"] or not current["deterministic"]:
            raise RuntimeError("recording supports deterministic decisions without an action mask")
        logits = np.concatenate([current["logits"][i] for i in range(self.head_count)], axis=-1)
        return {
            "obs": current["obs"],
            "rnn_in": current["rnn_in"],
            "masks": current["masks"],
            "gru_out": np.concatenate(current["gru_out"], axis=0),
            "act_in": current["act_in"],
            "logits": logits.astype(np.float32),
            "actions": current["actions"],
            "log_probs": current["log_probs"],
            "rnn_out": current["rnn_out"],
        }

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()


def aircraft_context(env: Any, uids: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Pre-decision simulator state of every aircraft (the state the observation came from)."""
    agents = [env.agents[uid] for uid in uids]
    return {
        "step": np.asarray(env.current_step, dtype=np.int64),
        "alive": np.asarray([a.is_alive for a in agents], dtype=bool),
        "shotdown": np.asarray([a.is_shotdown for a in agents], dtype=bool),
        "crash": np.asarray([a.is_crash for a in agents], dtype=bool),
        "geodetic": np.asarray([a.get_geodetic() for a in agents], dtype=np.float64),
        "position_neu": np.asarray([a.get_position() for a in agents], dtype=np.float64),
        "rpy": np.asarray([a.get_rpy() for a in agents], dtype=np.float64),
        "velocity_neu": np.asarray([a.get_velocity() for a in agents], dtype=np.float64),
        "missiles_left": np.asarray([a.num_missiles - len(a.launch_missiles) for a in agents], dtype=np.int64),
    }


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout.strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_everything(seed: int) -> None:
    """The exact global seeding of mtd-rl-lag ``evaluate_mhi_team.py`` (before the env exists)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def head_weights(actor: torch.nn.Module) -> dict[str, np.ndarray]:
    """The action heads' weights and biases stacked in logit order: logits = act_in @ weight.T + bias."""
    heads = [head.logits_net for head in actor.act.action_outs]
    weight = np.concatenate([h.weight.detach().cpu().numpy() for h in heads]).astype(np.float32)
    bias = np.concatenate([h.bias.detach().cpu().numpy() for h in heads]).astype(np.float32)
    return {"weight": weight, "bias": bias, "head_sizes": np.asarray([h.out_features for h in heads], dtype=np.int64)}


def record_episode(args: argparse.Namespace, seed: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from algorithms.utils.team_manifest import load_team_bundle
    from envs.JSBSim.envs import MultipleCombatEnv
    from scripts.eval.mhi_eval_metrics import MatchMetrics
    from scripts.eval.team_match_driver import TeamSide, play_team_match

    seed_everything(seed)
    env = MultipleCombatEnv(args.scenario)
    started = time.monotonic()
    try:
        env.enable_step_telemetry(False)
        if env.action_mask_layout() is not None:
            raise RuntimeError("scenario applies an action mask; the replay does not model it")
        bundle = load_team_bundle(env, args.checkpoint, args.policy_manifest)
        learners = tuple(env.control_plan.learner_ids)
        uids = tuple(env.control_plan.agent_ids)
        side = TeamSide(learners, bundle, args.protocol, deterministic=True)
        metrics = MatchMetrics(env)
        tap = ActorTap(bundle.policy.actor)
        rows: list[dict[str, np.ndarray]] = []

        def before_step(_context: Any) -> None:
            rows.append({**tap.take(), **aircraft_context(env, uids)})

        def after_step(context: Any) -> None:
            n_agents = len(uids)
            rows[-1]["reward"] = np.asarray(context.rewards, dtype=np.float64).reshape(n_agents, -1)
            rows[-1]["done"] = np.asarray(context.dones, dtype=bool).reshape(n_agents, -1)

        _obs, dones, _info = play_team_match(
            env, [side], seed, before_step=before_step, after_step=after_step, max_steps=args.max_steps
        )
        tap.remove()
        arrays = {name: np.stack([row[name] for row in rows]) for name in (*DECISION_FIELDS, *CONTEXT_FIELDS)}
        arrays["final_rnn_state"] = np.asarray(side.rnn_state, dtype=np.float32)
        np.savez(args.out / "head_weights.npz", **head_weights(bundle.policy.actor))  # same actor every seed
        summary = {
            "seed": seed,
            "file": episode_file(seed),
            "decisions": len(rows),
            "completed": dones is not None,
            "final_env_step": int(env.current_step),
            "wall_seconds": round(time.monotonic() - started, 3),
            "metrics": metrics.finish(),
            "sampled_initial_conditions": {uid: dict(agent.init_state) for uid, agent in env.agents.items()},
            "aircraft_uids": list(uids),
            "learner_uids": list(learners),
            "scripted_uids": list(env.control_plan.scripted_ids),
            "rule_type": env.control_plan.rule_type,
            "time_interval_s": float(env.time_interval),
            "world_profile": env.world_profile.to_dict(),
            "policy_spec": bundle.spec.to_dict(),
            "network_args": dict(vars(bundle.args)) if hasattr(bundle.args, "__dict__") else str(bundle.args),
        }
        return arrays, summary
    finally:
        env.close()


def provenance(args: argparse.Namespace, xavier_root: Path) -> dict[str, Any]:
    checkpoint = args.checkpoint.resolve()
    return {
        "lag_root": str(args.lag_root),
        "lag_git_commit": _git(args.lag_root, "rev-parse", "HEAD"),
        "lag_tracked_changes": _git(args.lag_root, "status", "--porcelain", "--untracked-files=no").splitlines(),
        "xavier_git_commit": _git(xavier_root, "rev-parse", "HEAD"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "policy_manifest": str(args.policy_manifest),
        "policy_manifest_sha256": sha256(args.policy_manifest),
        "dependencies": {name: importlib.metadata.version(name) for name in DEPENDENCIES},
        "python": sys.version.split()[0],
        "torch_threads": torch.get_num_threads(),
    }


def build_manifest(
    args: argparse.Namespace, xavier_root: Path, episodes: list[dict[str, Any]], shapes: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    first = episodes[0]
    info = provenance(args, xavier_root)
    actor_by_agent = {
        uid: {"actor_file": info["checkpoint"], "sha256": info["checkpoint_sha256"]} for uid in first["learner_uids"]
    }
    return {
        "schema": SCHEMA,
        "model_alias": args.model_alias,
        "scenario": args.scenario,
        "opponent": {"scripted_uids": first["scripted_uids"], "rule_type": first["rule_type"]},
        "protocol": args.protocol,
        "decision": "deterministic",
        "masks": "all_ones",
        "seeds": list(args.seeds),
        "max_steps": args.max_steps,
        "aircraft_uids": first["aircraft_uids"],
        "learner_uids": first["learner_uids"],
        "actor_by_agent": actor_by_agent,
        "action_head_sizes": first["policy_spec"].get("action_head_sizes"),
        "time_interval_s": first["time_interval_s"],
        "world_profile": first["world_profile"],
        "policy_spec": first["policy_spec"],
        "network_args": first["network_args"],
        "provenance": info,
        "array_shapes": shapes,
        "fields": {**DECISION_FIELDS, **CONTEXT_FIELDS, "final_rnn_state": FINAL_STATE_FIELD},
        "episodes": [
            {k: v for k, v in e.items() if k not in ("world_profile", "policy_spec", "network_args")} for e in episodes
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lag-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new, empty output directory")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--protocol", choices=PROTOCOLS, default=PROTOCOLS[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--max-steps", type=int, help="truncate episodes (default: run to termination)")
    parser.add_argument(
        "--policy-manifest", type=Path, help="default: LAG_ROOT/docs/mhi_integration/policy_manifest.json"
    )
    args = parser.parse_args(argv)
    args.lag_root = args.lag_root.resolve()
    args.out = args.out.resolve()
    if args.policy_manifest is None:
        args.policy_manifest = args.lag_root / "docs/mhi_integration/policy_manifest.json"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        print(f"refusing to overwrite non-empty {args.out}", file=sys.stderr)
        return 2
    sys.path.insert(0, str(args.lag_root))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    episodes: list[dict[str, Any]] = []
    shapes: dict[str, dict[str, Any]] = {}
    for seed in args.seeds:
        arrays, summary = record_episode(args, seed)
        np.savez_compressed(args.out / summary["file"], **arrays)
        shapes[summary["file"]] = {name: [list(a.shape), str(a.dtype)] for name, a in arrays.items()}
        episodes.append(summary)
        outcome = summary["metrics"]["outcome"]
        print(json.dumps({"seed": seed, "decisions": summary["decisions"], "outcome": outcome}), flush=True)
    manifest = build_manifest(args, Path(__file__).resolve().parents[1], episodes, shapes)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "episodes": len(episodes)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
