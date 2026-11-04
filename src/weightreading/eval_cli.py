"""``evaluate`` subcommand: argument registration and body (kept out of ``cli.py`` on purpose)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from weightreading.claims import ClaimsError
from weightreading.commands import DEFAULT_OUTPUT_ROOT
from weightreading.evaluate import EvaluationError, evaluate_run, load_settings, write_evaluation

EXIT_OK = 0
EXIT_REFUSED = 1


def add_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("evaluate", help="score a run's claims against the policy's measured behaviour (offline)")
    parser.add_argument("--run-dir", type=Path, required=True, help="outputs/weightreading/runs/<run> (read only)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="the checkpoint the run read")
    parser.add_argument("--spec", type=Path, required=True, help="spec with an `evaluation` section")
    parser.add_argument("--config", type=Path, help="optional YAML overriding evaluation settings (samples, seed, ...)")
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="writes <root>/evaluations/<run>/"
    )


def run(args: argparse.Namespace, emit: Callable[[object], None]) -> int:
    try:
        settings = load_settings(args.config)
        payload = evaluate_run(args.run_dir, args.checkpoint, args.spec, settings)
        directory = write_evaluation(payload, args.output_root / "evaluations" / args.run_dir.resolve().name)
    except (EvaluationError, ClaimsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    headline = {
        name: {
            "overall": block["summary"]["overall"]["accuracy"],
            "signed": block["summary"]["signed_claims"]["accuracy"],
            "coverage": block["coverage_hit_rate"],
        }
        for name, block in payload["conditions"].items()
    }
    emit({"evaluation_dir": str(directory), "claims": payload["claims_parsed"]["n"], "headline": headline})
    return EXIT_OK
