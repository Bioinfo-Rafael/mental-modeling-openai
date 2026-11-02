#!/usr/bin/env python3
"""Measured Decisions calls and offline replay. Only execute sends API requests."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from realtime_experiment import workflow as w


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--run", required=True)
    plan.add_argument("--repeats", type=int, default=1)
    plan.add_argument("--max-windows", type=int)
    for name in ("execute", "export", "status", "request"):
        command = commands.add_parser(name)
        command.add_argument("--run", required=True)
        if name == "execute":
            command.add_argument("--resume", action="store_true")
            command.add_argument("--limit", type=int, help="Maximum new API calls this invocation")
        if name == "request":
            command.add_argument("--query", help="Query ID; default is the first gate")
    args = parser.parse_args()
    if args.command == "plan":
        directory = w.create_plan(args.run, args.repeats, args.max_windows)
        w.export(directory)
        print(directory)
    else:
        directory = w.resolve_run(args.run)
        if args.command == "execute":
            try:
                w.execute(directory, resume=args.resume, limit=args.limit)
            finally:
                w.export(directory)
        elif args.command == "export":
            w.export(directory)
        elif args.command == "status":
            print(w.json.dumps(w.export(directory), ensure_ascii=False, indent=2))
        elif args.command == "request":
            manifest = w.verify(directory)
            query = next(q for q in manifest["queries"] if q["id"] == args.query) if args.query else manifest["queries"][0]
            print(w.json.dumps(w.request_for(directory, query), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from None
