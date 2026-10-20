"""Freeze five commentary windows and prepare or execute model-specific batches."""
import argparse
import hashlib
import json
from pathlib import Path
from lag_data.dataset import load_episode, PROCESSED
from .config import EPISODE, CENTERS, MODELS, REFERENCE_URL
from .dataset import build_window
from .prompts import build_messages, PROMPT_VERSION
from . import execution as ex, batch, evaluation
HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]


def make_plan(models=('sol', 'terra', 'luna'), repeats=1, variant='commentary'):
    if repeats < 1 or not models or len(set(models)) != len(models):
        raise ValueError('Require distinct models and positive repeats')
    resolved = [MODELS.get(m, m) for m in models]
    if len(set(resolved)) != len(resolved):
        raise ValueError('Models resolve to duplicate IDs')
    frame = load_episode(PROCESSED/f'{EPISODE}.csv')
    plan = dict(schema_version='1.0', prompt_version='lag-tactical-v1' if variant == 'tactical' else PROMPT_VERSION,
                prompt_variant=variant,
                request_options={'reasoning_effort': 'medium'}, reference_url=REFERENCE_URL,
                samples={}, queries=[], source_sha256={})
    for i, center in enumerate(CENTERS):
        window = build_window(frame, center)
        sid = f'center_{center:04d}'
        sample = dict(episode_id=EPISODE, target_id='all', task='commentary', observation_scope='all_aircraft_missiles',
                      H=61, sample_number=i, start=center-30, target_step=center,
                      window=window, messages=build_messages(window, variant))
        plan['samples'][sid] = sample
        for model_i, (alias, model) in enumerate(zip(models, resolved)):
            for repeat in range(1, repeats+1):
                q = dict(condition_id=f'{model}/commentary', sample_id=sid, model_alias=f'model{model_i}',
                         model=model, requested_model=alias, repeat_id=repeat)
                q['request_sha256'] = ex.digest(ex.request_for(plan, q))
                q['query_id'] = ex.digest(q)
                plan['queries'].append(q)
    plan['window_overlap'] = [dict(centers=[a,b], time_points=max(0, min(a+30,b+30)-max(a-30,b-30)+1))
                              for i,a in enumerate(CENTERS) for b in CENTERS[i+1:]]
    paths = [HERE/'run.py', PROCESSED/f'{EPISODE}.csv', PROCESSED/'manifest.json']
    paths += sorted((HERE/'commentary_experiment').glob('*.py')) + sorted((HERE/'commentary_experiment').glob('*.txt'))
    paths += sorted((HERE.parent/'13_LAG/lag_data').glob('*.py'))
    plan['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    plan['counts'] = dict(samples=5, requests=len(plan['queries']), models=len(models), repeats=repeats)
    plan['plan_id'] = ex.digest(plan)
    ex.verify_plan(plan)
    return plan


def run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    for mode in ('dry-run', 'execute', 'collect', 'status', 'export', 'adopt'):
        modes.add_argument('--'+mode, action='store_true')
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--repeats', type=int)
    parser.add_argument('--output')
    parser.add_argument('--prompt-variant', choices=('commentary', 'tactical'))
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--job')
    parser.add_argument('--batch-id')
    args = parser.parse_args(argv)
    if args.wait and not (args.execute or args.collect):
        parser.error('--wait requires --execute or --collect')
    if not 1 <= args.poll_seconds <= 60:
        parser.error('--poll-seconds must be 1..60')
    if args.adopt != bool(args.job and args.batch_id) or ((args.job or args.batch_id) and not args.adopt):
        parser.error('--adopt requires both --job and --batch-id')
    saved = args.collect or args.status or args.export or args.adopt
    if saved and (args.models is not None or args.repeats is not None or args.prompt_variant is not None):
        parser.error('Saved modes use the saved conditions')
    variant = args.prompt_variant or 'commentary'
    output = args.output or ('tactical_batch_v1' if variant == 'tactical' else 'commentary_batch_v1')
    directory = (HERE/'results'/output).resolve()
    if directory == (HERE/'results').resolve() or not directory.is_relative_to((HERE/'results').resolve()):
        parser.error('--output must stay inside results/')
    with ex.run_lock(directory):
        if saved:
            plan = ex.read_json(directory/'manifest.json')
            ex.verify_plan(plan)
            if args.status:
                print(json.dumps(batch.local_status(batch.load_state(plan, directory)), indent=2))
            elif args.export:
                evaluation.evaluate_saved(plan, directory)
            else:
                batch.execute(plan, directory, repository_root=ROOT, submit_new=False, wait=args.wait,
                              poll_seconds=args.poll_seconds, adopt_job=args.job, batch_id=args.batch_id)
        else:
            plan = make_plan(args.models or tuple(MODELS), args.repeats if args.repeats is not None else 1, variant)
            ex.save_plan(plan, directory)
            for sid, sample in plan['samples'].items():
                ex.write_json(directory/'windows'/f'{sid}.json', sample['window'])
            evaluation.evaluate_saved(plan, directory)
            state = batch.prepare(plan, directory)
            print(json.dumps(plan['counts']), '\nSaved:', directory)
            if args.execute:
                batch.execute(plan, directory, repository_root=ROOT, wait=args.wait, poll_seconds=args.poll_seconds)
            else:
                print(json.dumps(batch.local_status(state), indent=2))
                print('No API calls made.')
    return 0
