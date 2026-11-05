"""``intent-preview`` / ``intent-run`` subcommands (method B of the intent comparison).

``intent-run`` sends paid requests only with BOTH ``--execute`` and ``--confirm-paid-api`` (checked
in ``cli.main``); without them it writes the same local preview as ``intent-preview``. The API key
is read only after the first sandbox preflight passed.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mtd_rl_xavier.intent_compare import check_run_id
from weightreading import intent_execute, intent_run
from weightreading.commands import DEFAULT_OUTPUT_ROOT, utc_stamp
from weightreading.intent_run import EXIT_OK

COMMANDS = ("intent-preview", "intent-run")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--compare", type=Path, default=intent_run.DEFAULT_COMPARE, help="shared compare.yaml")
    parser.add_argument("--config", type=Path, default=intent_run.DEFAULT_CONFIG, help="intent-B run config")
    parser.add_argument("--spec", type=Path, default=intent_run.DEFAULT_SPEC)
    parser.add_argument("--checkpoint", type=Path, help="default: the actor named in the rollout manifest")
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="B artifacts go to <root>/intent/ (gitignored)"
    )


def add_parsers(sub: argparse._SubParsersAction) -> None:
    _add_common(sub.add_parser("intent-preview", help="method B: messages, inputs and cost bound (local)"))
    run = sub.add_parser("intent-run", help="method B paid run; a local preview unless --execute --confirm-paid-api")
    _add_common(run)
    run.add_argument(
        "--run-id", help="shared run id (outputs/intent_compare/<run_id>/); default seed<seed>, as method A"
    )
    run.add_argument("--resume", action="store_true", help="continue an interrupted run with the same --run-id")
    run.add_argument("--execute", action="store_true", help="send paid requests (also needs --confirm-paid-api)")
    run.add_argument("--confirm-paid-api", action="store_true")


def _summary(payload: dict) -> dict:
    sessions = [
        {
            "step": entry["step"],
            "aircraft": entry["aircraft"],
            "window": entry["window"],
            "skipped": entry["skipped"],
            **(
                {}
                if entry["skipped"]
                else {
                    "first_message_tokens_approx": entry["estimate"]["first_message_tokens_approx"],
                    "hard_cap_cost_usd": entry["estimate"]["hard_cap"]["cost_usd"],
                    "expected_cost_usd": entry["estimate"]["expected"]["cost_usd"],
                    "below_long_context_threshold": entry["estimate"]["hard_cap"]["below_long_context_threshold"],
                }
            ),
        }
        for entry in payload["sessions"]
    ]
    return {"trace_validation": payload["trace_validation"], "sessions": sessions, "totals": payload["totals"]}


def _preview(prepared: intent_run.IntentPrepared, root: Path, emit: Callable[[object], None]) -> int:
    directory = root / "intent" / "previews" / f"{utc_stamp()}_seed{prepared.compare.seed}"
    payload = intent_run.write_preview(prepared, directory)
    emit({"preview_dir": str(directory), **_summary(payload)})
    print("Preview only. API requests made: 0", file=sys.stderr)
    return EXIT_OK


def run(args: argparse.Namespace, emit: Callable[[object], None], client_factory: Callable[[], Any]) -> int:
    prepared = intent_run.prepare(args.compare, args.config, args.spec, args.checkpoint)
    if args.command == "intent-preview" or not args.execute:
        return _preview(prepared, args.output_root, emit)
    run_id = check_run_id(args.run_id or f"seed{prepared.compare.seed}")
    code, record = intent_execute.execute_run(
        prepared, run_id, client_factory, args.output_root / "intent", resume=args.resume
    )
    if code != EXIT_OK:
        emit({"run_id": run_id, "status": record.get("status")})
        print("Refused: sandbox preflight failed. API requests made: 0", file=sys.stderr)
        return code
    answered = sum(1 for item in record["sessions"] if item.get("answered"))
    emit({"run_id": run_id, "cues": record["cues_path"], "sessions": len(record["sessions"]), "answered": answered})
    return EXIT_OK
