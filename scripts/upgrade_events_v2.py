#!/usr/bin/env python3
"""Convert a translation run of event schema 2 (before 2026-11-08) into a new schema-3 run directory, once.

    .venv/bin/python scripts/upgrade_events_v2.py --run outputs/translator/<run> --out outputs/translate/<new> \
        --map initial=start --map start=change --map change=change --map request=request

Schema 2 had the trigger kinds initial, start, change and request; schema 3 has start, change and
request with the 2026-11-09 meanings. No kind is guessed: every kind the run uses needs a --map.
The old directory is never modified; request payloads and their request_sha256 are kept as they are.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

KINDS = ("start", "change", "request")
RUN_FILES = ("events.jsonl", "requests.jsonl", "responses.jsonl")


def upgrade(record: dict[str, Any], kinds: dict[str, str]) -> dict[str, Any]:
    upgraded = {**record, "schema_version": 3}
    if "trigger_reasons" in record:
        reasons = [{**reason, "kind": kinds[reason["kind"]]} for reason in record["trigger_reasons"]]
        upgraded = {**upgraded, "trigger_reasons": reasons}
    if "primary_trigger" in record:
        upgraded = {**upgraded, "primary_trigger": kinds[record["primary_trigger"]]}
    return upgraded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True, help="schema-2 run directory (read only)")
    parser.add_argument("--out", type=Path, required=True, help="new run directory (must not exist)")
    parser.add_argument("--map", action="append", default=[], help="OLD=NEW, NEW one of start, change, request")
    args = parser.parse_args(argv)
    kinds = dict(item.split("=", 1) for item in args.map)
    if not set(kinds.values()) <= set(KINDS):
        parser.error(f"every --map target must be one of {KINDS}")
    records = {}
    for name in RUN_FILES:
        lines = (args.run / name).read_text(encoding="utf-8").splitlines()
        records = {**records, name: [json.loads(line) for line in lines if line]}
    if any(record.get("schema_version") != 2 for record in records["events.jsonl"]):
        parser.error("--run must be a run of event schema 2")
    used = {reason["kind"] for record in records["events.jsonl"] for reason in record.get("trigger_reasons", [])}
    if used - set(kinds):
        parser.error(f"no --map for the trigger kinds {sorted(used - set(kinds))}")
    args.out.mkdir(mode=0o700, parents=True, exist_ok=False)
    shutil.copy2(args.run / "run.json", args.out / "run.json")
    for name, items in records.items():
        lines = [json.dumps(upgrade(item, kinds), ensure_ascii=False) + "\n" for item in items]
        (args.out / name).write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "events": len(records["events.jsonl"]), "map": kinds}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
