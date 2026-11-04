"""Freeze baseline provenance, exact prompts, labels and requests before sending."""
import difflib
import hashlib
import json
from .common import HERE, ROOT, ex, batch, prompt_tools, lag
from .dataset import build_samples


VARIANTS = ('foresight_short_v1', 'targets_v1')


def prompt_spec(variant='foresight_short_v1'):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown prompt variant: {variant}')
    folder = HERE / 'prompts' / variant
    comparison = ex.read_json(folder / 'changes/provenance.json')
    comparison.update(baseline_templates={}, diffs={})
    templates = {}
    for role in ('system', 'user'):
        current = (folder / f'{role}.txt').read_text()
        baseline = (folder / f'changes/baseline_{role}.txt').read_text()
        if prompt_tools.text_sha256(baseline) != comparison['baseline_sha256'][role]:
            raise ValueError(f'Baseline {role} hash differs')
        if prompt_tools.text_sha256(current) != comparison['proposal_sha256'][role]:
            raise ValueError(f'Proposal {role} hash differs; version the prompt and provenance')
        diff = ''.join(difflib.unified_diff(
            baseline.splitlines(True), current.splitlines(True),
            fromfile=f"{comparison['source_run']}/{role}.txt", tofile=f'{variant}/{role}.txt'))
        if diff != (folder / f'changes/{role}.diff').read_text():
            raise ValueError(f'Stale {role} diff')
        templates[role] = dict(content=current, sha256=prompt_tools.text_sha256(current),
                               path=str((folder / f'{role}.txt').relative_to(ROOT)))
        comparison['baseline_templates'][role] = baseline
        comparison['diffs'][role] = diff
    comparison['change_notes'] = (folder / 'changes/README.md').read_text()
    return dict(version=variant.replace('_', '-'), variant=variant, templates=templates,
                output_format='aircraft_intentions_json', comparison=comparison,
                description='固定30遷移の目的と動きのみ。n=30は後続情報込みの参照回答。',
                length=dict(max_words=80 if variant == 'targets_v1' else 55,
                            target_sentences=[1, 2] if variant == 'targets_v1' else [1, 1],
                            max_sentences=2 if variant == 'targets_v1' else 1))


def messages(window, spec):
    data = '\n'.join(json.dumps(block, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
                     for block in window['blocks'])
    fields = {k: v for k, v in window.items() if k != 'blocks'}
    return [dict(role='system', content=spec['templates']['system']['content']),
            dict(role='user', content=spec['templates']['user']['content'].format(**fields, window_data=data))]


def make_plan(variant='foresight_short_v1'):
    config = ex.read_json(HERE / 'configs/cutoff_sweep_v1.json')
    annotations = ex.read_json(HERE / 'data/annotations_v1.json')
    if (config['history_before'], config['horizon'], config['repeats'], config['offsets']) != (30, 30, 1, [-1, 1, 30]):
        raise ValueError('v1 requires history=30, horizon=30, no repeats, offsets=[-1,1,30]')
    if config['boundary_policy'] != 'exclude_event_from_all_conditions':
        raise ValueError('Unsupported boundary policy')
    if annotations['episode_id'] != config['episode_id']:
        raise ValueError('Annotation episode differs')
    ids = [e['id'] for e in annotations['events']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate annotation ID')
    spec = prompt_spec(variant)
    config['prompt_version'] = variant
    if config['models'] != spec['comparison']['models'] or config['request_options'] != spec['comparison']['request_options']:
        raise ValueError('v1 retains the saved v2 models and request options')
    csv = lag.PROCESSED / f"{config['episode_id']}.csv"
    samples, excluded = build_samples(lag.load_episode(csv), annotations, config)
    plan = dict(schema_version='1.0', experiment='20_foresight_LAG', config=config,
                prompt_version=spec['version'], prompt_variant=spec['variant'], prompt_spec=spec,
                request_options=config['request_options'], samples=samples, queries=[],
                annotations=annotations, excluded_events=excluded,
                reference_url=None, reference_kind='hindsight_model_output_not_ground_truth')
    for sid, sample in samples.items():
        sample['messages'] = messages(sample['window'], spec)
        for index, model in enumerate(config['models']):
            q = dict(sample_id=sid, model=model, requested_model=model, model_alias=f'model{index}',
                     repeat_id=1, condition_id=f"{model}/n{sample['offset']:+d}")
            q['request_sha256'] = ex.digest(ex.request_for(plan, q))
            q['query_id'] = ex.digest(q)
            plan['queries'].append(q)
    paths = [p for p in HERE.rglob('*') if p.is_file() and
             not set(p.relative_to(HERE).parts) & {'results', 'tests', '__pycache__', '.pytest_cache'}]
    paths += [csv, csv.parent / 'manifest.json']
    paths += [ROOT / 'experiments/13_LAG/lag_data' / name for name in ('dataset.py', 'definitions.py', '__init__.py')]
    paths += [ROOT / 'experiments/18_desicion_LAG/decision_experiment/dataset.py']
    paths += [ROOT / 'experiments/17_commentary_LAG/commentary_experiment' / name
              for name in ('execution.py', 'batch.py', 'evaluation.py', 'prompts.py', 'config.py', '__init__.py')]
    plan['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(set(paths))}
    plan['counts'] = dict(events=len(samples) // 3, excluded_events=len(excluded), samples=len(samples),
                          models=len(config['models']), input_conditions=3,
                          model_conditions=3 * len(config['models']), repeats=1, requests=len(plan['queries']))
    plan['plan_id'] = ex.digest(plan)
    ex.verify_plan(plan)
    return plan


def save_plan(plan, directory):
    ex.save_plan(plan, directory)
    ex.write_json(directory / 'annotations.json', plan['annotations'])
    ex.write_json(directory / 'excluded_events.json', plan['excluded_events'])
    for sid, sample in plan['samples'].items():
        ex.write_json(directory / 'windows' / f'{sid}.json', sample['window'])
    # Keep the exact code used by this run alongside its hashes.
    for relative in plan['source_sha256']:
        if relative.endswith('.py'):
            dest = directory / 'code' / relative
            content = (ROOT / relative).read_bytes()
            if dest.exists() and dest.read_bytes() != content:
                raise ValueError(f'Code snapshot differs: {relative}')
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
    batch.prepare(plan, directory)
