import argparse
import json
from .common import ROOT, ex, batch, run_directory
from .planning import make_plan, save_plan
from .evaluation import export
from .viewer import build_viewer


def run(argv=None):
    parser = argparse.ArgumentParser(description='Fixed 30-transition forecasts; plan is offline.')
    parser.add_argument('command', choices=['plan', 'execute', 'collect', 'status', 'export', 'adopt'])
    parser.add_argument('--run', default='foresight_short_v1')
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--job')
    parser.add_argument('--batch-id')
    args = parser.parse_args(argv)
    if args.wait and args.command not in ('execute', 'collect'):
        parser.error('--wait is only valid with execute/collect')
    if not 1 <= args.poll_seconds <= 60:
        parser.error('--poll-seconds must be 1..60')
    if args.command == 'adopt':
        if not args.job or not args.batch_id:
            parser.error('adopt requires --job and --batch-id')
    elif args.job or args.batch_id:
        parser.error('--job/--batch-id require adopt')
    directory = run_directory(args.run)
    if args.command != 'plan' and not (directory / 'manifest.json').exists():
        parser.error('No saved run; use plan first')
    with ex.run_lock(directory):
        if args.command == 'plan':
            plan = make_plan()
            save_plan(plan, directory)
        else:
            plan = ex.read_json(directory / 'manifest.json')
            ex.verify_plan(plan)
        try:
            if args.command in ('execute', 'collect', 'adopt'):
                batch.execute(plan, directory, repository_root=ROOT,
                              submit_new=args.command == 'execute', wait=args.wait,
                              poll_seconds=args.poll_seconds, adopt_job=args.job, batch_id=args.batch_id)
        finally:
            rows, summary = export(plan, directory)
            build_viewer(plan, rows, summary, directory)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print('Viewer:', directory / 'report/review.html')
        if args.command not in ('execute', 'collect', 'adopt'):
            print('No API calls made.')
    return 0
