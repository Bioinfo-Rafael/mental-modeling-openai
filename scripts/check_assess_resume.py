#!/usr/bin/env python3
"""Offline check that an existing assess journal still covers every request the judge would send.

Rebuilds the assess requests of one Translator run exactly like ``evaluation.assess_cli judge`` and
checks that each ``request_sha256`` is already journaled, so the next ``judge --execute`` re-sends
nothing. Builds no client, reads no key and writes no file.

    python scripts/check_assess_resume.py --translator-run outputs/translator/<run> \
        --journal outputs/evaluation/assess/<run>/judge_requests.jsonl [--expect <request_sha256> ...]

Exit codes: 0 every planned request is journaled (and every --expect hash is planned); 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluation.assess_checks import check_generation_inputs, invalid_targets
from evaluation.assess_cli import _episode_frame
from evaluation.assess_data import DEFAULT_CONFIG, load_config, load_run, target_key
from evaluation.assess_judge import plan_calls
from mtd_rl_xavier.lag.dataset import load_episode

EXIT_OK, EXIT_MISMATCH = 0, 1


def planned_hashes(translator_run: Path, config_path: Path) -> list[str]:
    run = load_run(translator_run)
    config = load_config(config_path)
    invalid = invalid_targets(check_generation_inputs(run, config))
    events = [event for event in run.events if target_key(event["target"]) not in invalid]
    calls = plan_calls(
        events, run=run, config=config, annotation=None, load_frame=lambda: _episode_frame(run, load_episode)
    )
    return [call.request_sha256 for call in calls]


def journaled_hashes(journal: Path) -> set[str]:
    lines = journal.read_text(encoding="utf-8").splitlines()
    return {json.loads(line)["request_sha256"] for line in lines if line.strip()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--translator-run", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--expect", action="append", default=[], help="a request_sha256 that must be planned")
    args = parser.parse_args(argv)

    planned = planned_hashes(args.translator_run, args.config)
    sent = journaled_hashes(args.journal)
    missing = [digest for digest in planned if digest not in sent]
    unexpected = [digest for digest in args.expect if digest not in planned]
    ok = bool(planned) and not missing and not unexpected
    print(json.dumps({"ok": ok, "planned": planned, "not_journaled": missing, "expect_not_planned": unexpected}))
    return EXIT_OK if ok else EXIT_MISMATCH


if __name__ == "__main__":
    sys.exit(main())
