"""Exp.05/06/03_1 targets with Exp.10 ICR-FQI and Exp.09 reward ablation."""
import argparse
from dataclasses import asdict, dataclass
from importlib import import_module
from itertools import islice, product
from pathlib import Path

from experiments import common
from experiments.joint_questions import JOINT_QUESTIONS
from experiments.split_execution import execution_lock
from preprocessing.llmx_original import build_prompt_queries, discover_episodes
from llm_x.data import Episode

icrl = import_module('experiments.10_ICRL.runner')
prompts = import_module('experiments.10_ICRL.prompts')
without_rewards = import_module('experiments.09_ablation_reward.runner.prompts').without_rewards
scoring = icrl.scoring
HERE = Path(__file__).resolve().parent
ROOT = common.ROOT
IDENTITY_FIELDS = ('pair_id', 'model_alias', 'model', 'reward_present', 'reward_mode',
                   'ordinal', 'episode_path', 'history_start', 'history_end_exclusive', 'query_index')
CSV_FIELDS = tuple(dict.fromkeys(icrl.CSV_FIELDS + IDENTITY_FIELDS))


@dataclass(frozen=True)
class Experiment:
    models: tuple[str, ...]
    reward_modes: tuple[bool, ...]
    task: str
    metric: str
    H: int
    n: int
    method: str


def make_plan(spec):
    if (spec.task != 'Pendulum-v1' or spec.metric != 'next-action'
            or spec.H != 20 or spec.n != 10 or spec.method != 'icrfqi_k5'):
        raise ValueError('Expected Pendulum next-action, H20, N10, icrfqi_k5')
    if (len(spec.models) != 2 or set(spec.models) != {'terra', 'luna'}
            or len(spec.reward_modes) != 2 or set(spec.reward_modes) != {True, False}
            or any(type(value) is not bool for value in spec.reward_modes)):
        raise ValueError('Expected terra/luna and both boolean reward modes, without duplicates')
    sources = sorted((s for s in discover_episodes() if s.task == spec.task),
                     key=lambda s: s.path.as_posix())
    if not sources or sources[0].path.name != 'episode_0.npz':
        raise ValueError('Expected path-sorted first Pendulum episode to be episode_0.npz')
    source = sources[0]
    question = JOINT_QUESTIONS[spec.task][spec.metric]
    targets = list(islice(build_prompt_queries(source, metric=spec.metric,
        question_name=question, history_size=common.history_size(spec.H)), spec.n))
    if len(targets) != spec.n or any(
            (q.history_start, q.history_end, q.query_index) != (i, i + spec.H, i + spec.H)
            for i, q in enumerate(targets)):
        raise ValueError('Expected the first ten H20 windows: starts 0..9, queries 20..29')
    episode = Episode.load(source.path)
    if any(q.dataset_sha256 != episode.sha256 for q in targets):
        raise ValueError('Target episode changed during planning')
    episode_path = str(source.path.relative_to(ROOT))
    hashes = {episode_path: episode.sha256}
    rows = []
    for alias, reward, (ordinal, q) in product(spec.models, spec.reward_modes, enumerate(targets)):
        target = q.user_prompt if reward else without_rewards(q)
        user = prompts.compose_user(target, '', spec.method)  # Exp.10 E=1: no references.
        mode = 'with_reward' if reward else 'without_reward'
        truth = scoring.score(dict(task=spec.task, metric=spec.metric, query_index=q.query_index),
                              '', episode)
        pair = dict(model_alias=alias, ordinal=ordinal, episode_path=episode_path,
                    history_start=q.history_start, history_end_exclusive=q.history_end,
                    query_index=q.query_index)
        row = dict(**pair, pair_id=common.digest(common.json_text(pair)),
                   condition_id=f'{alias}__{spec.method}__{mode}', model=common.MODELS[alias],
                   task=spec.task, metric=spec.metric, H=spec.H, method=spec.method,
                   K=prompts.METHOD_K[spec.method], E=1, context_episode_count=1,
                   target_episode=source.path.name, episode_sha256=episode.sha256,
                   auxiliary_episode_paths=[], context_episode_sha256=hashes,
                   reward_present=reward, reward_mode=mode, reward_scope='target_history_only',
                   question_name=question, ground_truth=truth['ground_truth'],
                   action_value_ground_truth=truth['action_value_ground_truth'],
                   system_prompt=q.system_prompt, user_prompt=user, target_user_prompt=target,
                   base_target_user_prompt_sha256=common.digest(q.user_prompt),
                   system_prompt_sha256=common.digest(q.system_prompt), user_prompt_sha256=common.digest(user))
        row['request'] = common.expected_api_request(row['model'], q.system_prompt, user)
        row['query_id'] = common.digest(common.json_text(row))
        rows.append(row)
    if len({r['query_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate execution identities')
    return dict(experiment='11_ICRL_n10', status='ready', config=asdict(spec),
                input_sha256=hashes, queries=rows)


def save_preview(manifest, root):
    directory = root / 'dry_run'
    saved = directory / 'manifest.json'
    if saved.exists() and common.json_text(common.read_json(saved)) != common.json_text(common.redact(manifest)):
        raise FileExistsError('Existing dry run differs; archive results/dry_run before regenerating')
    common.write_json(saved, manifest)
    fields = ('query_id', 'condition_id', *IDENTITY_FIELDS, 'task', 'metric', 'H', 'method', 'K',
              'E', 'target_episode', 'episode_sha256', 'question_name', 'ground_truth',
              'action_value_ground_truth', 'base_target_user_prompt_sha256',
              'system_prompt_sha256', 'user_prompt_sha256')
    common.write_csv(directory / 'manifest.csv', [{k: q[k] for k in fields} for q in manifest['queries']])
    for q in manifest['queries']:
        common.write_text(directory / 'prompts' / f"{q['condition_id']}__query{q['ordinal']:02d}.txt",
                          '[SYSTEM]\n' + q['system_prompt'] + '\n\n[USER]\n' + q['user_prompt'])
    print(f"Planned {len(manifest['queries'])} requests. Preview: {directory}")


def summarize(root):
    """Keep Exp.10 metrics/denominators; attach reward and paired target identities."""
    metrics = icrl.summarize(root)
    manifest = common.read_json(root / 'api/manifest.json')
    queries = {q['query_id']: q for q in manifest['queries']}
    conditions = {q['condition_id']: q for q in queries.values()}
    for metric in metrics:
        q = conditions[metric['condition_id']]
        metric.update(reward_present=q['reward_present'], reward_mode=q['reward_mode'])
    output = root / 'analysis'
    common.write_json(output / 'summary.json', dict(conditions=metrics,
        note='Bin accuracy uses parsed bins; MAE uses parsed numeric actions. Token/time aggregates '
             'use saved scored responses. Each condition has ten distinct targets; pair_id joins '
             'with_reward/without_reward within a model. Windows overlap in episode_0.'))
    common.write_csv(output / 'conditions.csv', metrics)
    per_query = output / 'per_query.csv'
    if per_query.exists():
        import csv
        with per_query.open(encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row.update({key: queries[row['query_id']][key] for key in IDENTITY_FIELDS})
        common.write_csv(per_query, rows)
    return metrics


def run(spec, argv=None):
    parser = argparse.ArgumentParser(description='No flags: execute the 40-query reward ablation and summarize.')
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
                                     source_experiment='11_ICRL_n10', csv_fields=CSV_FIELDS)
        finally:
            if (root / 'api/manifest.json').exists():
                summarize(root)
