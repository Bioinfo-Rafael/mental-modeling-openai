#!/usr/bin/env python3
"""Offline check that existing method B runs could still be resumed after a repository change.

``python -m weightreading intent-run --resume`` refuses a run unless the recorded ``RESUME_KEYS`` of its
``run.json`` equal the current ones; ``inputs`` holds the compare, config and spec paths as strings
(relative, or absolute under the checkout) plus their SHA-256. This checker replays that comparison
for the repository files without loading a checkpoint, rollout or client:

* every recorded repository path, re-rooted from ``--recorded-root`` onto this checkout, still exists
  under the same relative path with the recorded SHA-256;
* method B's default spec and config still live at those relative paths;
* the prompt version is unchanged.

Absolute recorded paths keep matching after a merge only while the original checkout stays where it
is. Writes nothing.

    python scripts/check_method_b_resume.py --runs <checkout>/outputs/weightreading/intent/runs \
        --recorded-root <checkout>

Exit codes: 0 every run is compatible; 1 a run is not (details on stdout as JSON).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from mtd_rl_xavier import paths
from weightreading.intent_execute import RESUME_KEYS
from weightreading.intent_prompt import PROMPT_VERSION
from weightreading.intent_run import DEFAULT_CONFIG, DEFAULT_SPEC
from weightreading.provenance import sha256_file

EXIT_OK, EXIT_MISMATCH = 0, 1
REPO_INPUTS = ("compare", "config", "spec")  # the inputs that live in this repository
# RESUME_KEYS as recorded by the existing runs; a change would refuse every one of them.
EXPECTED_RESUME_KEYS = (
    "inputs",
    "prompt_version",
    "model",
    "reasoning_effort",
    "limits",
    "steps",
    "targets",
    "window_steps",
)


def relative_to_repo(recorded: str, recorded_root: Path) -> Path:
    path = Path(recorded)
    return path.relative_to(recorded_root) if path.is_absolute() else path


def run_problems(run_json: dict[str, Any], recorded_root: Path) -> list[str]:
    problems = []
    if run_json.get("prompt_version") != PROMPT_VERSION:
        problems.append(f"prompt_version {run_json.get('prompt_version')!r} != {PROMPT_VERSION!r}")
    inputs = run_json.get("inputs", {})
    for name in REPO_INPUTS:
        relative = relative_to_repo(inputs[f"{name}_path"], recorded_root)
        current = paths.ROOT / relative
        if not current.is_file():
            problems.append(f"{name}: {relative} is missing")
        elif sha256_file(current) != inputs[f"{name}_sha256"]:
            problems.append(f"{name}: {relative} content differs")
    return problems


def default_problems(recorded_root: Path, run_json: dict[str, Any]) -> list[str]:
    """A resume uses the CLI defaults unless overridden: they must still name the recorded files."""
    inputs = run_json.get("inputs", {})
    problems = []
    for name, default in (("spec", DEFAULT_SPEC), ("config", DEFAULT_CONFIG)):
        recorded = relative_to_repo(inputs[f"{name}_path"], recorded_root)
        if default.relative_to(paths.ROOT) != recorded:
            problems.append(f"default {name} moved: {default.relative_to(paths.ROOT)} != {recorded}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True, help="outputs/weightreading/intent/runs of a checkout")
    parser.add_argument("--recorded-root", type=Path, required=True, help="the checkout the runs were made in")
    args = parser.parse_args(argv)

    report: dict[str, list[str]] = {}
    if tuple(RESUME_KEYS) != EXPECTED_RESUME_KEYS:
        report["RESUME_KEYS"] = [f"changed: {RESUME_KEYS}"]
    for run_file in sorted(args.runs.glob("*/run.json")):
        run_json = json.loads(run_file.read_text(encoding="utf-8"))
        problems = run_problems(run_json, args.recorded_root) + default_problems(args.recorded_root, run_json)
        report[run_file.parent.name] = problems
    ok = bool(report) and not any(report.values())
    print(json.dumps({"ok": ok, "runs": report}, ensure_ascii=False))
    return EXIT_OK if ok else EXIT_MISMATCH


if __name__ == "__main__":
    sys.exit(main())
