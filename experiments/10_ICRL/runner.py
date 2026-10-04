"""Plan the fixed ICRL grid; reuse Exp.09 execution and Exp.08 Joint scoring."""
import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from importlib import import_module
from itertools import product
from pathlib import Path
from statistics import mean

from experiments import common
from experiments.joint_questions import JOINT_QUESTIONS
from experiments.split_execution import execution_lock
from preprocessing.llmx_original import build_prompt_queries, discover_episodes
from llm_x.data import Episode
from .prompts import METHOD_K, compose_user, reference_prefix

HERE = Path(__file__).resolve().parent
ROOT = common.ROOT
scoring = import_module('experiments.08_fewshot.runner.scoring')
CSV_FIELDS = tuple(dict.fromkeys(scoring.CSV_FIELDS + (
    'model', 'method', 'K', 'E', 'context_episode_count', 'repeat_id', 'history_start',
    'query_index', 'target_episode', 'reward_present', 'element_accuracy',
    'parse_error', 'action_value_parse_error',
)))


@dataclass(frozen=True)
class Experiment:
    models: tuple[str, ...]
    methods: tuple[str, ...]
    context_episode_counts: tuple[int, ...]
    repeats: int
    task: str
    metric: str
    H: int
    start: int
    target_episode: str
    reward_present: bool


def make_plan(spec):
    if (spec.task != 'Pendulum-v1' or spec.metric != 'next-action'
            or spec.H != 20 or spec.start != 0 or spec.reward_present is not True
            or spec.target_episode != 'episode_9_seed3407.npz'):
        raise ValueError('Expected Pendulum next-action, H20, t0, episode 9, rewards present')
    for axis, allowed in ((spec.models, {'terra', 'luna'}),
                          (spec.methods, METHOD_K), (spec.context_episode_counts, {1, 3, 9})):
        if not axis or len(set(axis)) != len(axis) or not set(axis) <= set(allowed):
            raise ValueError('Invalid or duplicate experiment axis')
    if type(spec.repeats) is not int or spec.repeats < 1:
        raise ValueError('Repeats must be a positive integer')

    # Select exclusively by path before loading any action/reward values.
    sources = sorted((s for s in discover_episodes() if s.task == spec.task),
                     key=lambda s: s.path.as_posix())
    targets = [s for s in sources if s.path.name == spec.target_episode]
    if len(targets) != 1:
        raise ValueError('Expected exactly one target episode')
    target = targets[0]
    auxiliaries = [s for s in sources if s.path != target.path]
    required = max(spec.context_episode_counts) - 1
    if len(auxiliaries) < required:
        raise ValueError(f'Need {required} auxiliary episodes, found {len(auxiliaries)}')
    selected = [target, *auxiliaries[:required]]
    episodes = {s.path: Episode.load(s.path) for s in selected}
    hashes = {str(s.path.relative_to(ROOT)): episodes[s.path].sha256 for s in selected}
    question = JOINT_QUESTIONS[spec.task][spec.metric]
    q = next((q for q in build_prompt_queries(
        target, metric=spec.metric, question_name=question, history_size=common.history_size(spec.H)
    ) if q.history_start == spec.start), None)
    if q is None or q.history_end != spec.start + spec.H or q.query_index != q.history_end:
        raise ValueError('Missing or unexpected target window')
    if q.dataset_sha256 != episodes[target.path].sha256:
        raise ValueError('Target episode changed during planning')
    references = {E: reference_prefix([episodes[s.path] for s in auxiliaries[:E-1]],
                                      start=spec.start, H=spec.H)
                  for E in spec.context_episode_counts}
    truth = scoring.score(dict(task=spec.task, metric=spec.metric, query_index=q.query_index),
                          '', episodes[target.path])
    rows = []
    for alias, method, E, repeat in product(spec.models, spec.methods,
                                           spec.context_episode_counts, range(1, spec.repeats + 1)):
        auxiliary_paths = [str(s.path.relative_to(ROOT)) for s in auxiliaries[:E-1]]
        episode_path = str(target.path.relative_to(ROOT))
        user = compose_user(q.user_prompt, references[E], method)
        condition = f'{alias}__{method}__E{E}'
        row = dict(condition_id=condition, model_alias=alias, model=common.MODELS[alias],
                   method=method, K=METHOD_K[method], E=E, context_episode_count=E,
                   task=spec.task, metric=spec.metric, H=spec.H, history_start=spec.start,
                   history_end_exclusive=q.history_end, query_index=q.query_index,
                   target_episode=spec.target_episode, episode_path=episode_path,
                   episode_sha256=q.dataset_sha256, auxiliary_episode_paths=auxiliary_paths,
                   context_episode_sha256={p: hashes[p] for p in [episode_path, *auxiliary_paths]},
                   reward_present=True, repeat_id=repeat, ordinal=repeat-1,
                   question_name=question, ground_truth=truth['ground_truth'],
                   action_value_ground_truth=truth['action_value_ground_truth'],
                   system_prompt=q.system_prompt, user_prompt=user, target_user_prompt=q.user_prompt,
                   system_prompt_sha256=common.digest(q.system_prompt), user_prompt_sha256=common.digest(user))
        row['request'] = common.expected_api_request(row['model'], q.system_prompt, user)
        row['query_id'] = common.digest(common.json_text(row))
        rows.append(row)
    if len({r['query_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate execution identities')
    return dict(experiment='10_ICRL', status='ready', config=asdict(spec),
                input_sha256=hashes, queries=rows)


def save_preview(manifest, root):
    directory = root / 'dry_run'
    saved = directory / 'manifest.json'
    if saved.exists() and common.json_text(common.read_json(saved)) != common.json_text(common.redact(manifest)):
        raise FileExistsError('Existing dry run differs; archive results/dry_run before regenerating')
    common.write_json(saved, manifest)
    fields = ('query_id', 'condition_id', 'model_alias', 'model', 'method', 'K', 'E',
              'context_episode_count', 'H', 'history_start', 'query_index', 'target_episode',
              'auxiliary_episode_paths', 'context_episode_sha256', 'repeat_id')
    common.write_csv(directory / 'manifest.csv', [{k: q[k] for k in fields} for q in manifest['queries']])
    for q in manifest['queries']:
        common.write_text(directory / 'prompts' / f"{q['condition_id']}__repeat{q['repeat_id']}.txt",
                          '[SYSTEM]\n' + q['system_prompt'] + '\n\n[USER]\n' + q['user_prompt'])
    print(f"Planned {len(manifest['queries'])} requests. Preview: {directory}")


def summarize(root, *, csv_fields=CSV_FIELDS,
              condition_fields=('model_alias', 'model', 'method', 'K', 'E', 'context_episode_count'),
              note='Bin accuracy uses parsed bins; MAE uses parsed numeric actions. Token/time aggregates use saved scored responses. Repeats share one target query.'):
    """Aggregate the existing scorer's outputs, with explicit parse denominators."""
    manifest = common.read_json(root / 'api/manifest.json')
    records_path = root / 'api/records.jsonl'
    records = common.read_jsonl(records_path) if records_path.exists() else []
    requests_path = root / 'api/requests.jsonl'
    attempted = {r['query_id'] for r in common.read_jsonl(requests_path)} if requests_path.exists() else set()
    derived = []
    for r in records:
        bin_ok = r['status'] in ('match', 'mismatch')
        action_ok = r.get('absolute_error') is not None
        derived.append({**{k: r.get(k) for k in csv_fields}, 'bin_parse_success': bin_ok,
                        'action_parse_success': action_ok, 'parse_success': bin_ok and action_ok})
    lookup = {r['query_id']: r for r in derived}
    groups = defaultdict(list)
    for q in manifest['queries']:
        groups[q['condition_id']].append(q)
    metrics = []
    for condition, queries in groups.items():
        results = [lookup[q['query_id']] for q in queries if q['query_id'] in lookup]
        valid = [r for r in results if r['bin_parse_success']]
        errors = [r['absolute_error'] for r in results if r['action_parse_success']]
        matches = sum(r['status'] == 'match' for r in valid)
        q = queries[0]
        metric = dict(condition_id=condition, **{k: q[k] for k in condition_fields},
                      planned=len(queries), responses=len(results), parsed_bins=len(valid),
                      parsed_actions=len(errors), parse_successes=sum(r['parse_success'] for r in results),
                      matches=matches, bin_accuracy=matches/len(valid) if valid else None,
                      match_rate_all_planned=matches/len(queries),
                      action_mae=mean(errors) if errors else None,
                      unattempted=sum(q['query_id'] not in attempted for q in queries),
                      unresolved=sum(q['query_id'] in attempted and q['query_id'] not in lookup for q in queries))
        for field in ('input_tokens', 'output_tokens', 'total_tokens', 'request_elapsed_seconds'):
            values = [r[field] for r in results if r.get(field) is not None]
            metric[field + '_sum'] = sum(values) if values else None
            metric[field + '_mean'] = mean(values) if values else None
            metric[field + '_n'] = len(values)
        metrics.append(metric)
    output = root / 'analysis'
    common.write_json(output / 'summary.json', dict(conditions=metrics, note=note))
    common.write_csv(output / 'conditions.csv', metrics)
    if derived:
        common.write_csv(output / 'per_query.csv', derived)
    return metrics


def run(spec, argv=None):
    parser = argparse.ArgumentParser(description='No flags: execute the ICRL grid and summarize.')
    parser.add_argument('--dry-run', action='store_true', help='Generate prompts without API calls')
    parser.add_argument('--resume', action='store_true', help='Continue only unattempted requests')
    args = parser.parse_args(argv)
    if args.dry_run and args.resume:
        parser.error('--dry-run and --resume cannot be combined')
    manifest = make_plan(spec)
    root = HERE / 'results'
    with execution_lock(root):
        if not args.dry_run and (root / 'api').exists() and not args.resume:
            raise FileExistsError('Existing API run: use --resume; no requests sent')
        save_preview(manifest, root)
        if args.dry_run:
            return 0
        execution = import_module('experiments.09_ablation_reward.runner.execution')
        try:
            return execution.execute(manifest, root, resume=args.resume,
                                     source_experiment='10_ICRL', csv_fields=CSV_FIELDS)
        finally:
            if (root / 'api/manifest.json').exists():
                summarize(root)
