"""``python -m weightreading <command>``.

Commands: ``inspect``, ``validate-spec``, ``bundle``, ``preview``, ``run``, ``audit``,
``sandbox-selftest``. Nothing here calls a paid API unless ``run`` gets BOTH ``--execute`` and
``--confirm-paid-api``; without them ``run`` writes the same local preview as ``preview``. The
API key is read from the environment only after both flags and the sandbox preflight pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mtd_rl_xavier import paths
from weightreading import commands, eval_cli
from weightreading.audit import write_audit
from weightreading.backends import make_openai_client
from weightreading.bundle import BundleError
from weightreading.config import TASK_MODES, ConfigError, load_config
from weightreading.sandbox import SandboxError
from weightreading.spec import SpecError
from weightreading.torch_reader import CheckpointError

DEFAULT_CONFIG = paths.CONFIGS / "weightreading" / "run.yaml"
EXIT_OK = 0
EXIT_REFUSED = 1


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True, help="torch zip checkpoint (state_dict)")
    parser.add_argument("--spec", type=Path, required=True, help="configs/weightreading/*.spec.json")


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    _add_inputs(parser)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="run configuration YAML")
    parser.add_argument("--task-mode", choices=TASK_MODES, help="override the config's task_mode")
    parser.add_argument("--output-root", type=Path, default=commands.DEFAULT_OUTPUT_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m weightreading", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="list tensors of a checkpoint (restricted reader, no torch)")
    inspect.add_argument("checkpoint", type=Path)
    inspect.add_argument("--tensor-hashes", action="store_true", help="print {key: sha256 of raw bytes} only")

    _add_inputs(sub.add_parser("validate-spec", help="check a spec against a checkpoint"))

    bundle = sub.add_parser("bundle", help="write the LLM bundle (weights as .npy + manifest)")
    _add_inputs(bundle)
    bundle.add_argument("--output-root", type=Path, default=commands.DEFAULT_OUTPUT_ROOT)

    _add_run_options(sub.add_parser("preview", help="bundle + first message + token/cost preview (local)"))

    run = sub.add_parser("run", help="paid session; a local preview unless --execute --confirm-paid-api")
    _add_run_options(run)
    run.add_argument("--execute", action="store_true", help="send paid requests (also needs --confirm-paid-api)")
    run.add_argument("--confirm-paid-api", action="store_true")

    audit = sub.add_parser("audit", help="flag forward-pass attempts in a run's tool_calls.jsonl")
    audit.add_argument("run_dir", type=Path)

    selftest = sub.add_parser("sandbox-selftest", help="run the fail-closed sandbox preflight")
    selftest.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    selftest.add_argument("--bundle", type=Path, help="existing bundle dir (default: a tiny throwaway bundle)")
    eval_cli.add_parser(sub)
    return parser


def _preview(args: argparse.Namespace) -> int:
    config = commands.with_task_mode(load_config(args.config, paths.ROOT), args.task_mode)
    prepared = commands.prepare(args.checkpoint, args.spec, args.output_root)
    name = commands.output_name(prepared, config)
    directory = args.output_root / "previews" / name
    payload = commands.write_preview(prepared, config, directory)
    _print({"preview_dir": str(directory), **payload})
    print("Preview only. API requests made: 0", file=sys.stderr)
    return EXIT_OK


def _run(args: argparse.Namespace, client_factory: Callable[[], Any]) -> int:
    if not args.execute:
        return _preview(args)
    config = commands.with_task_mode(load_config(args.config, paths.ROOT), args.task_mode)
    prepared = commands.prepare(args.checkpoint, args.spec, args.output_root)
    name = commands.output_name(prepared, config)
    run_dir = args.output_root / "runs" / name
    code, record = commands.execute_run(prepared, config, run_dir, client_factory)
    if code != EXIT_OK:
        _print({"run_dir": str(run_dir), "status": record.get("status"), "preflight": record["sandbox_preflight"]})
        print("Refused: sandbox preflight failed. API requests made: 0", file=sys.stderr)
        return code
    audit_path = write_audit(run_dir)
    _print({"run_dir": str(run_dir), "session": record["session"], "audit": str(audit_path)})
    return EXIT_OK


def _dispatch(args: argparse.Namespace, client_factory: Callable[[], Any]) -> int:
    if args.command == "inspect":
        checkpoint = commands.read_checkpoint(args.checkpoint)
        _print(commands.inspect_payload(checkpoint, args.tensor_hashes))
        return EXIT_OK
    if args.command == "validate-spec":
        checkpoint, spec = commands.load_checked(args.checkpoint, args.spec)
        _print({"ok": True, "network_alias": spec.alias, "tensors": len(checkpoint.tensors)})
        return EXIT_OK
    if args.command == "bundle":
        prepared = commands.prepare(args.checkpoint, args.spec, args.output_root)
        _print({"bundle": str(prepared.bundle.path), "manifest_sha256": prepared.bundle.manifest_sha256})
        return EXIT_OK
    if args.command == "preview":
        return _preview(args)
    if args.command == "run":
        return _run(args, client_factory)
    if args.command == "audit":
        _print({"audit": str(write_audit(args.run_dir))})
        return EXIT_OK
    if args.command == "evaluate":
        return eval_cli.run(args, _print)
    result = commands.sandbox_selftest(load_config(args.config, paths.ROOT), args.bundle)
    _print(result.to_dict())
    return EXIT_OK if result.ok else EXIT_REFUSED


def main(argv: list[str] | None = None, client_factory: Callable[[], Any] = make_openai_client) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run" and args.execute != args.confirm_paid_api:
        parser.error("Paid execution requires BOTH --execute and --confirm-paid-api")
    try:
        return _dispatch(args, client_factory)
    except (CheckpointError, SpecError, ConfigError, BundleError, SandboxError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_REFUSED
