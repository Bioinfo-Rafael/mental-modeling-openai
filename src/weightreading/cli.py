"""``python -m weightreading <command>``: method B of the intent comparison.

Commands: ``sandbox-selftest``, ``intent-preview``, ``intent-run``. Nothing here calls a paid API
unless ``intent-run`` gets BOTH ``--execute`` and ``--confirm-paid-api``; without them it writes the
same local preview as ``intent-preview``. The API key is read from the environment only after both
flags and the sandbox preflight pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mtd_rl_xavier.intent_compare import CompareConfigError
from translate import paths
from translate.llm.openai_responses import make_openai_client
from weightreading import commands, intent_cli, intent_run
from weightreading.bundle import BundleError
from weightreading.config import ConfigError, load_config
from weightreading.intent_episode import EpisodeError
from weightreading.intent_execute import PreflightRefused
from weightreading.intent_protocol import IntentAnswerError
from weightreading.intent_trace import TraceError
from weightreading.sandbox import SandboxError
from weightreading.spec import SpecError
from weightreading.torch_reader import CheckpointError

DEFAULT_CONFIG = intent_run.DEFAULT_CONFIG  # configs/weightreading/intent_b.yaml (its sandbox section)
EXIT_OK = 0
EXIT_REFUSED = 1


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m weightreading", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    selftest = sub.add_parser("sandbox-selftest", help="fail-closed sandbox preflight (no API call)")
    selftest.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    selftest.add_argument("--bundle", type=Path, help="existing bundle dir (default: a tiny throwaway bundle)")
    intent_cli.add_parsers(sub)
    return parser


def _dispatch(args: argparse.Namespace, client_factory: Callable[[], Any]) -> int:
    if args.command in intent_cli.COMMANDS:
        return intent_cli.run(args, _print, client_factory)
    result = commands.sandbox_selftest(load_config(args.config, paths.ROOT), args.bundle)
    _print(result.to_dict())
    return EXIT_OK if result.ok else EXIT_REFUSED


def main(argv: list[str] | None = None, client_factory: Callable[[], Any] = make_openai_client) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "intent-run" and args.execute != args.confirm_paid_api:
        parser.error("Paid execution requires BOTH --execute and --confirm-paid-api")
    try:
        return _dispatch(args, client_factory)
    except (CheckpointError, SpecError, ConfigError, BundleError, SandboxError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except (CompareConfigError, EpisodeError, TraceError, IntentAnswerError, PreflightRefused) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_REFUSED
