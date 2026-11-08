#!/usr/bin/env python3
"""Replay one recorded rollout seed with ACMI output on and prove it is the same rollout.

Run with mtd-rl-lag's interpreter (torch + JSBSim) on a compute node, via
``slurm/episode_prepare.sbatch``. The setup is the recorder's
(``policy_record_rollouts.py``): seeds before the env exists, one torch thread, pinned actor
through ``load_team_bundle``, deterministic ``TeamSide``, the real ``play_team_match`` loop, and
the recorder's own ``aircraft_context`` read before every decision. The only difference is
``acmi_path``. Every captured aircraft field must be bitwise equal to the npz, otherwise the
script exits 2 and nothing downstream (rendering, conversion) may use the ACMI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from policy_record_rollouts import aircraft_context, seed_everything, sha256

COMPARED_FIELDS = (
    "step",
    "alive",
    "shotdown",
    "crash",
    "geodetic",
    "position_neu",
    "rpy",
    "velocity_neu",
    "missiles_left",
)
EXIT_MISMATCH = 2


def replay(args: argparse.Namespace) -> dict[str, np.ndarray]:
    from algorithms.utils.team_manifest import load_team_bundle
    from envs.JSBSim.envs import MultipleCombatEnv
    from scripts.eval.team_match_driver import TeamSide, play_team_match

    seed_everything(args.seed)
    env = MultipleCombatEnv(args.scenario)
    try:
        env.enable_step_telemetry(False)
        bundle = load_team_bundle(env, args.checkpoint, args.policy_manifest)
        uids = tuple(env.control_plan.agent_ids)
        side = TeamSide(tuple(env.control_plan.learner_ids), bundle, args.protocol, deterministic=True)
        rows: list[dict[str, np.ndarray]] = []

        def before_step(_context: Any) -> None:
            rows.append(aircraft_context(env, uids))

        play_team_match(env, [side], args.seed, before_step=before_step, acmi_path=args.acmi_out)
        return {name: np.stack([row[name] for row in rows]) for name in COMPARED_FIELDS}
    finally:
        env.close()


def compare(captured: dict[str, np.ndarray], reference: Any) -> dict[str, bool]:
    """``np.array_equal`` per field, shape and dtype included (bitwise for float64)."""
    result: dict[str, bool] = {}
    for name in COMPARED_FIELDS:
        ref = reference[name]
        same = captured[name].shape == ref.shape and captured[name].dtype == ref.dtype
        result = {**result, name: bool(same and np.array_equal(captured[name], ref))}
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lag-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--npz", type=Path, required=True, help="recorded episode_seed<seed>.npz")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--acmi-out", type=Path, required=True, help="new ACMI file (inside LAG LOCAL_ONLY/)")
    parser.add_argument("--report", type=Path, required=True, help="JSON report path")
    parser.add_argument("--scenario", default="2v2/ShootMissile/HierarchyVsbaseline_mhi_v1_3")
    parser.add_argument("--protocol", default="per_agent_v1")
    parser.add_argument("--policy-manifest", type=Path)
    args = parser.parse_args(argv)
    args.lag_root = args.lag_root.resolve()
    if args.policy_manifest is None:
        args.policy_manifest = args.lag_root / "docs/mhi_integration/policy_manifest.json"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.acmi_out.exists():
        print(f"refusing to overwrite {args.acmi_out}", file=sys.stderr)
        return 1
    sys.path.insert(0, str(args.lag_root))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.acmi_out.parent.mkdir(parents=True, exist_ok=True)
    captured = replay(args)
    with np.load(args.npz) as reference:
        equal = compare(captured, reference)
        reference_rows = int(reference["step"].shape[0])
    report = {
        "seed": args.seed,
        "npz": str(args.npz),
        "npz_sha256": sha256(args.npz),
        "acmi": str(args.acmi_out),
        "acmi_sha256": sha256(args.acmi_out),
        "checkpoint_sha256": sha256(args.checkpoint),
        "replay_rows": int(captured["step"].shape[0]),
        "reference_rows": reference_rows,
        "bitwise_equal": equal,
        "all_equal": all(equal.values()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_equal": report["all_equal"], "rows": report["replay_rows"]}))
    return 0 if report["all_equal"] else EXIT_MISMATCH


if __name__ == "__main__":
    sys.exit(main())
