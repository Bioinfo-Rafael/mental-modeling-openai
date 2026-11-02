#!/usr/bin/env python3
"""Generate A1/A2 prose for positive saved gates; execution is synchronous and timed."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from realtime_experiment import narrative as n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('plan', 'execute', 'export', 'request', 'status'):
        p = sub.add_parser(name)
        p.add_argument('--run', required=True)
        if name == 'plan':
            p.add_argument('--source-run', default='pilot_20261026_v2')
        if name == 'execute':
            p.add_argument('--resume', action='store_true'); p.add_argument('--limit', type=int)
            p.add_argument('--retry-failed', action='store_true', help='With --resume, retry recorded failures once; preserve all attempts and skip successes/uncertain requests')
        if name == 'request':
            p.add_argument('--query')
    args = parser.parse_args()
    if args.command == 'plan':
        print(n.create_plan(args.run, args.source_run)); return
    directory = n.w.resolve_run(args.run)
    if args.command == 'execute':
        n.execute(directory, resume=args.resume, retry_failed=args.retry_failed, limit=args.limit)
    elif args.command == 'export':
        print(n.w.json.dumps(n.export(directory), indent=2))
    elif args.command == 'status':
        print(n.w.json.dumps(n.w.read_json(directory / 'report/summary.json'), indent=2))
    else:
        manifest = n.w.verify(directory)
        q = next(q for q in manifest['queries'] if q['id'] == args.query) if args.query else manifest['queries'][0]
        print(n.w.json.dumps(n.request_for(directory, q), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from None
