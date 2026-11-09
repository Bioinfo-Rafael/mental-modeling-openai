"""``python -m evaluation``: judge a translation run, import human ratings, write the report.

    python -m evaluation judge --run DIR [--config C] [--name N] [--execute --confirm-paid-api [--only-event ID]]
    python -m evaluation import-ratings --run DIR --ratings FILE [--config C] [--name N]
    python -m evaluation report --run DIR [--config C] [--name N]

Every subcommand works in ``<output>/<translation run>/<name>/`` (owner-only, reused). ``judge``
previews unless both paid flags are given; a paid run holds the marker, scans every journal under
the output root and never sends a known request, except the one stuck event named by ``--only-event``
(an explicit, user-requested re-send). A paid run from a git worktree is refused. Exit codes: 0 ok or preview; 1 failure; 2 usage or
gate; 3 a call has an uncertain outcome (it may have been billed; never resend it); 4 the call cap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.audit import audit_run, leaking_targets
from evaluation.config import JudgeConfig, criteria_for, load_config
from evaluation.judge import (
    GUARD_ROOT,
    Call,
    journaled,
    known_requests,
    main_checkout_only,
    make_plan,
    missing_results,
    paid_marker,
    resend_plan,
    run_calls,
)
from evaluation.material import event_kinds, load_frame, load_run, read_jsonl, target_key
from evaluation.prompts import build_prompt
from evaluation.results import RatingsError, build_report, import_ratings
from translate.backends import BackendError
from translate.backends.openai import OpenAIBackend, make_client
from translate.journal import REPO_ROOT, append_jsonl, repo_path, request_sha256

DEFAULT_CONFIG = Path("configs/evaluation/judge.yaml")
EXIT_CODES = {"completed": 0, "preview": 0, "failed": 1, "uncertain": 3, "cap_reached": 4}
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
DIR_MODE = 0o700  # judge material holds scenario text and ratings: owner-only
CHECKOUT = REPO_ROOT  # a paid run must start from the main checkout, never a git worktree

ClientFactory = Callable[..., Any]


def _name(value: str) -> str:
    if not SAFE_NAME.fullmatch(value):
        raise argparse.ArgumentTypeError("must be a name (letters, digits, . _ -), never a path")
    return value


def _event_id(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("must be an event id")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evaluation", description=__doc__.split("\n\n")[0])
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run", type=Path, required=True, help="translation run directory")
    common.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="judge config (repo-relative)")
    common.add_argument("--name", type=_name, help="judgement directory name (default: the config's file stem)")
    commands = parser.add_subparsers(dest="command", required=True)
    judge = commands.add_parser("judge", parents=[common], help="preview, or with both flags the paid judge")
    judge.add_argument("--execute", action="store_true", help="send the judge calls (needs --confirm-paid-api)")
    judge.add_argument("--confirm-paid-api", action="store_true", help="confirm paid calls (needs --execute)")
    judge.add_argument(
        "--only-event",
        type=_event_id,
        metavar="EVENT_ID",
        help="re-send one stuck event (sent from this directory, no verdict here); needs both paid flags",
    )
    ratings = commands.add_parser("import-ratings", parents=[common], help="validate and append human ratings")
    ratings.add_argument("--ratings", type=Path, required=True, help="human ratings JSONL (private/)")
    commands.add_parser("report", parents=[common], help="write report.json")
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _directory(args: argparse.Namespace, config: JudgeConfig, run: Any) -> Path:
    """``<output>/<run>/<name>/``, bound to the run's file hashes on first use."""
    directory = repo_path(config.output) / run.path.name / (args.name or args.config.stem)
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(mode=DIR_MODE, exist_ok=True)
    previous = directory / "run.json"
    if previous.is_file() and json.loads(previous.read_text(encoding="utf-8"))["run_sha256"] != run.file_sha256:
        raise ValueError(f"{run.path} changed since {directory} was first used: judge it under another --name")
    return directory


def _write_run_json(directory: Path, args: argparse.Namespace, config: JudgeConfig, run: Any, **fields: Any) -> None:
    record = {"translation_run": str(run.path), "run_sha256": run.file_sha256, "config": str(args.config)}
    record = {**record, "config_sha256": _sha256(repo_path(args.config)), "resolved": config.model_dump(mode="json")}
    record = {**record, "prompt_sources": config.prompt_sources}
    (directory / "run.json").write_text(
        json.dumps({**record, **fields}, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _calls(run: Any, config: JudgeConfig, directory: Path) -> tuple[list[Call], set[str]]:
    """The audit (written to audit.jsonl) and one planned call per event of a target that did not leak."""
    audit = audit_run(run)
    (directory / "audit.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in audit), "utf-8")
    leaks = leaking_targets(audit)
    frame = load_frame(run)
    calls = []
    for event in run.events:
        if target_key(event["target"]) in leaks:
            continue
        criteria = criteria_for(config, event_kinds(event))
        prompt = build_prompt(event, run, frame, config, criteria)
        body = OpenAIBackend.request_body(config.model, prompt)
        calls.append(Call(event, criteria, body, request_sha256(body)))
    return calls, leaks


def _send_planned(
    plan: Any, client_factory: ClientFactory, directory: Path, config: JudgeConfig, note: dict | None
) -> Any:
    """Build the paid client (after the gate, the marker and the scan) and send the planned calls."""
    backend = OpenAIBackend(config.model, client=client_factory(timeout_s=config.model.timeout_s))
    batch = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return run_calls(
        plan, send=lambda call: backend.send(call.body), directory=directory, config=config, batch=batch, note=note
    )


def run_judge(args: argparse.Namespace, config: JudgeConfig, run: Any, client_factory: ClientFactory) -> int:
    directory = _directory(args, config, run)
    calls, leaks = _calls(run, config, directory)
    roots = (GUARD_ROOT, repo_path(config.output))
    if not args.execute:
        plan = make_plan(calls, known_requests(*roots), journaled(directory))
        preview = {"plan": plan.to_json(), "requests": [{"event_id": c.event_id, "body": c.body} for c in calls]}
        (directory / "preview.json").write_text(json.dumps(preview, indent=1, ensure_ascii=False), encoding="utf-8")
        _write_run_json(directory, args, config, run, status="preview", plan=plan.to_json(), leaks=sorted(leaks))
        _print({"judgement_dir": str(directory), "status": "preview", "plan": plan.to_json(), "leaks": sorted(leaks)})
        print("Preview only. API requests made: 0", file=sys.stderr)
        return 0
    main_checkout_only(CHECKOUT)
    with paid_marker(GUARD_ROOT):  # one paid judge run at a time, whatever `output` says
        if args.only_event is not None:  # the explicit, user-requested re-send of one stuck event; nothing else
            plan, note = resend_plan(calls, args.only_event, directory, roots)
        else:
            plan, note = (
                make_plan(calls, known_requests(*roots), journaled(directory)),
                None,
            )  # scanned under the marker
        if plan.refused:
            raise ValueError(f"{len(plan.refused)} event(s) were judged here with other bytes: use another --name")
        only = {"only_event": args.only_event}
        _write_run_json(
            directory, args, config, run, status="running", plan=plan.to_json(), leaks=sorted(leaks), **only
        )
        try:
            outcome = _send_planned(plan, client_factory, directory, config, note) if plan.to_send else None
        except (Exception, KeyboardInterrupt) as exc:  # the client could not be built: nothing was sent
            _write_run_json(
                directory, args, config, run, status="failed", plan=plan.to_json(), error=type(exc).__name__, **only
            )
            raise
        status = outcome.status if outcome else "completed"
        counts = (
            {"attempted": outcome.attempted, "judged": outcome.judged} if outcome else {"attempted": 0, "judged": 0}
        )
        error = outcome.error if outcome else None
        missing = missing_results(plan, directory)
        if status == "completed" and missing:  # sent before, but no verdict here: never resent, so say so
            status, error = (
                "failed",
                {"message": f"{len(missing)} sent event(s) have no results here", "first": missing[0]},
            )
        _write_run_json(
            directory, args, config, run, status=status, plan=plan.to_json(), counts=counts, error=error, **only
        )
    _print(
        {"judgement_dir": str(directory), "status": status, "plan": plan.to_json(), "counts": counts, "error": error}
    )
    return EXIT_CODES[status]


def run_import(args: argparse.Namespace, config: JudgeConfig, run: Any) -> int:
    directory = _directory(args, config, run)
    path = directory / "ratings.jsonl"
    existing = read_jsonl(path) if path.is_file() else []
    records = import_ratings(read_jsonl(args.ratings), run, config, existing)
    imported = {"import_sha256": _sha256(args.ratings)}
    if records:
        append_jsonl(path, *({**record, **imported} for record in records))
    _print({"judgement_dir": str(directory), "ratings_imported": len(records)})
    return 0


def run_report(args: argparse.Namespace, config: JudgeConfig, run: Any) -> int:
    directory = _directory(args, config, run)
    results = read_jsonl(directory / "results.jsonl") if (directory / "results.jsonl").is_file() else []
    ratings = read_jsonl(directory / "ratings.jsonl") if (directory / "ratings.jsonl").is_file() else []
    report = build_report(run, config, results, ratings, leaking_targets(audit_run(run)))
    (directory / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _print({"judgement_dir": str(directory), "report": str(directory / "report.json")})
    return 0


def main(argv: list[str] | None = None, *, client_factory: ClientFactory = make_client) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "judge" and args.execute != args.confirm_paid_api:
        parser.error("the judge calls a paid API: give BOTH --execute and --confirm-paid-api, or neither")
    if args.command == "judge" and args.only_event is not None and not args.execute:
        parser.error("--only-event re-sends a paid request: give --execute --confirm-paid-api with it")
    try:
        config = load_config(repo_path(args.config))
        run = load_run(args.run)
        if args.command == "judge":
            return run_judge(args, config, run, client_factory)
        if args.command == "import-ratings":
            return run_import(args, config, run)
        return run_report(args, config, run)
    except (ValueError, OSError, RatingsError, BackendError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
