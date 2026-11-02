"""Readable commands: plan offline, execute explicitly, review, clone, compare."""
import argparse
import json

from .evaluation import compare_runs, export_run
from .execution import execute
from .planning import create_plan, request_for, verify_plan
from .prompts import CONDITIONS, clone_prompt
from .storage import RESULTS, resolve_run


def parser():
    result = argparse.ArgumentParser(description="LAG explanation-trigger experiments (gpt-6-luna)")
    commands = result.add_subparsers(dest="command", required=True)
    for stage in ("develop", "evaluate"):
        command = commands.add_parser(stage, help="Create a frozen offline plan; sends no API requests")
        command.add_argument("--prompt", default="v001")
        command.add_argument("--run", required=True, help="Unique run name within stage/prompt")
        command.add_argument("--history", type=int, choices=(30, 40))
        command.add_argument("--stride", type=int)
        command.add_argument("--repeats", type=int)
        command.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high", "xhigh", "max"))
        command.add_argument("--max-output-tokens", type=int)
        command.add_argument("--timeout-seconds", type=int)
        command.add_argument("--step-tolerance", type=int)
        command.add_argument("--order-seed", type=int)
        if stage == "evaluate":
            command.add_argument("--conditions", nargs="+", choices=CONDITIONS)
    command = commands.add_parser("execute", help="Send the saved requests to paid APIs")
    command.add_argument("--run", required=True)
    command.add_argument("--resume", action="store_true")
    command.add_argument("--limit", type=int, help="Send at most N never-started requests")
    for name in ("export", "status"):
        command = commands.add_parser(name)
        command.add_argument("--run", required=True)
    command = commands.add_parser("request", help="Print one exact saved request without sending it")
    command.add_argument("--run", required=True)
    command.add_argument("--query", help="Query ID; defaults to the first query")
    command = commands.add_parser("clone-prompt")
    command.add_argument("--from", dest="source", required=True)
    command.add_argument("--version", required=True)
    command.add_argument("--note", required=True)
    command = commands.add_parser("compare")
    command.add_argument("--runs", nargs="+", required=True)
    commands.add_parser("list", help="List runs by stage, prompt and run name")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command in ("develop", "evaluate"):
        overrides = {k: v for k, v in vars(args).items() if k not in ("command", "prompt", "run")}
        directory = create_plan(args.command, args.prompt, args.run, overrides)
        export_run(directory)
        manifest = verify_plan(directory)
        print(f'Offline plan: {directory}\nQueries: {len(manifest["queries"])} (no API calls)')
        print(f'Review: {directory / "report/review.html"}')
    elif args.command == "execute":
        directory = resolve_run(args.run)
        try:
            result = execute(directory, resume=args.resume, limit=args.limit)
        finally:
            if (directory / "manifest.json").exists():
                export_run(directory)
        print(json.dumps(result, indent=2))
    elif args.command in ("export", "status"):
        summary = export_run(resolve_run(args.run))
        print(json.dumps(summary["conditions"], indent=2))
    elif args.command == "request":
        directory = resolve_run(args.run)
        manifest = verify_plan(directory)
        query = next((q for q in manifest["queries"] if q["id"] == args.query), None) if args.query else manifest["queries"][0]
        if query is None:
            raise ValueError(f"Unknown query: {args.query}")
        print(json.dumps(request_for(directory, manifest, query), ensure_ascii=False, indent=2))
    elif args.command == "clone-prompt":
        print(clone_prompt(args.source, args.version, args.note))
    elif args.command == "compare":
        print(compare_runs([resolve_run(value) for value in args.runs]))
    elif args.command == "list":
        for path in sorted(RESULTS.glob("*/*/*/manifest.json")):
            manifest = verify_plan(path.parent)
            print(f'{path.parent.relative_to(RESULTS)}  {manifest["prompt_sha256"][:12]}  {len(manifest["queries"])} queries')
