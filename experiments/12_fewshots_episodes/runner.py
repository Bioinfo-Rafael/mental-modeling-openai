"""Pair the original N10 targets; reuse Exp.09's snapshot/executor and Exp.10 context."""
import argparse
from dataclasses import asdict, dataclass
from importlib import import_module
from itertools import islice, product
from pathlib import Path

from experiments import common
from experiments.joint_questions import JOINT_QUESTIONS
from experiments.split_execution import execution_lock
from preprocessing.llmx_original import build_prompt_queries, discover_episodes, file_sha256
from llm_x.data import Episode

icrl = import_module('experiments.10_ICRL.runner')
prompts = import_module('experiments.10_ICRL.prompts')
scoring = import_module('experiments.08_fewshot.runner.scoring')
HERE = Path(__file__).resolve().parent
ROOT = common.ROOT
SNAPSHOT = ROOT / 'experiments/09_ablation_reward/data_prep/fewshot_p1_k4.json'
IDENTITY_FIELDS = ('pair_id', 'model', 'context_method', 'episode_path', 'episode_sha256',
                   'history_start', 'history_end_exclusive', 'query_index')
CSV_FIELDS = tuple(dict.fromkeys(scoring.CSV_FIELDS + IDENTITY_FIELDS + (
    'element_accuracy', 'parse_error', 'action_value_parse_error', 'raw_response',
)))


@dataclass(frozen=True)
class Experiment:
    models: tuple[str, ...]
    context_methods: tuple[str, ...]
    task: str
    metric: str
    H: int
    n: int


def make_plan(spec):
    if (spec.task != 'Pendulum-v1' or spec.metric != 'next-action'
            or spec.H != 20 or spec.n != 10):
        raise ValueError('Expected Pendulum next-action, H20, N10')
    if (len(spec.models) != 2 or set(spec.models) != {'terra', 'luna'}
            or len(spec.context_methods) != 2
            or set(spec.context_methods) != {'fewshot4', 'episodes_E9'}):
        raise ValueError('Expected terra/luna × fewshot4/episodes_E9, without duplicates')
    snapshot = common.read_json(SNAPSHOT)
    prefix = snapshot['prefix']
    if common.digest(prefix) != snapshot['prefix_sha256']:
        raise ValueError('Frozen few-shot prefix changed')
    if (snapshot['shots'] != 4 or snapshot['score_pattern'] != 'pattern1'
            or len(snapshot['example_ids']) != 4 or len(set(snapshot['example_ids'])) != 4):
        raise ValueError('Expected frozen pattern1/k4 snapshot')

    # Select only by path, before inspecting episode actions or rewards.
    sources = sorted((s for s in discover_episodes() if s.task == spec.task),
                     key=lambda s: s.path.as_posix())
    targets = [s for s in sources if s.path.name == 'episode_0.npz']
    if len(targets) != 1:
        raise ValueError('Expected exactly one target episode_0.npz')
    source = targets[0]
    auxiliaries = [s for s in sources if s.path != source.path][:8]
    if len(auxiliaries) != 8:
        raise ValueError('Need eight auxiliary episodes for E9')
    episodes = {s.path: Episode.load(s.path) for s in [source, *auxiliaries]}
    hashes = {str(path.relative_to(ROOT)): e.sha256 for path, e in episodes.items()}
    episode_path = str(source.path.relative_to(ROOT))
    auxiliary_paths = [str(s.path.relative_to(ROOT)) for s in auxiliaries]
    question = JOINT_QUESTIONS[spec.task][spec.metric]
    queries = list(islice(build_prompt_queries(source, metric=spec.metric,
        question_name=question, history_size=common.history_size(spec.H)), spec.n))
    if len(queries) != spec.n or any(
            (q.history_start, q.history_end, q.query_index) != (i, i + spec.H, i + spec.H)
            for i, q in enumerate(queries)):
        raise ValueError('Expected the first ten H20 windows: starts 0..9, queries 20..29')
    if any(q.dataset_sha256 != episodes[source.path].sha256 for q in queries):
        raise ValueError('Target episode changed during planning')
    references = {q.history_start: prompts.reference_prefix(
        [episodes[s.path] for s in auxiliaries], start=q.history_start, H=spec.H) for q in queries}
    rows = []
    for alias, context, (ordinal, q) in product(spec.models, spec.context_methods, enumerate(queries)):
        fewshot = context == 'fewshot4'
        user = prefix + q.user_prompt if fewshot else prompts.compose_user(
            q.user_prompt, references[q.history_start], 'direct')
        truth = scoring.score(dict(task=spec.task, metric=spec.metric, query_index=q.query_index),
                              '', episodes[source.path])
        pair = dict(model_alias=alias, ordinal=ordinal, episode_path=episode_path,
                    history_start=q.history_start, history_end_exclusive=q.history_end,
                    query_index=q.query_index)
        row = dict(**pair, pair_id=common.digest(common.json_text(pair)),
                   condition_id=f'{alias}__{context}', model=common.MODELS[alias],
                   context_method=context, task=spec.task, metric=spec.metric, H=spec.H,
                   target_episode=source.path.name, episode_sha256=q.dataset_sha256,
                   # E describes raw trajectory context; demonstrations are recorded separately.
                   E=1 if fewshot else 9, context_episode_count=1 if fewshot else 9,
                   auxiliary_episode_paths=[] if fewshot else auxiliary_paths,
                   context_episode_sha256={episode_path: q.dataset_sha256} if fewshot else hashes,
                   auxiliary_windows=[] if fewshot else [dict(episode_path=p,
                       history_start=q.history_start, history_end_exclusive=q.history_end,
                       final_state_index=q.query_index) for p in auxiliary_paths],
                   shots=4 if fewshot else 0, score_pattern='pattern1' if fewshot else 'none',
                   example_ids=snapshot['example_ids'] if fewshot else [],
                   fewshot_prefix_sha256=snapshot['prefix_sha256'] if fewshot else None,
                   fewshot_source=str(SNAPSHOT.relative_to(ROOT)) if fewshot else None,
                   question_name=question, ground_truth=truth['ground_truth'],
                   action_value_ground_truth=truth['action_value_ground_truth'],
                   system_prompt=q.system_prompt, user_prompt=user, target_user_prompt=q.user_prompt,
                   target_offset=len(user) - len(q.user_prompt),
                   target_user_prompt_sha256=common.digest(q.user_prompt),
                   system_prompt_sha256=common.digest(q.system_prompt), user_prompt_sha256=common.digest(user))
        row['request'] = common.expected_api_request(row['model'], q.system_prompt, user)
        row['query_id'] = common.digest(common.json_text(row))
        rows.append(row)
    if len({r['query_id'] for r in rows}) != 40:
        raise ValueError('Expected 40 unique execution identities')
    return dict(experiment='12_fewshots_episodes', status='ready', config=asdict(spec),
                fewshot_provenance={k: v for k, v in snapshot.items() if k != 'prefix'},
                input_sha256={**hashes, str(SNAPSHOT.relative_to(ROOT)): file_sha256(SNAPSHOT)},
                queries=rows)


def save_preview(manifest, root):
    directory = root / 'dry_run'
    saved = directory / 'manifest.json'
    if saved.exists() and common.json_text(common.read_json(saved)) != common.json_text(common.redact(manifest)):
        raise FileExistsError('Existing dry run differs; archive results/dry_run before regenerating')
    common.write_json(saved, manifest)
    fields = ('query_id', 'condition_id', 'model_alias', 'ordinal', *IDENTITY_FIELDS,
              'task', 'metric', 'H', 'question_name', 'ground_truth', 'action_value_ground_truth',
              'shots', 'score_pattern', 'example_ids', 'fewshot_source', 'fewshot_prefix_sha256',
              'E', 'context_episode_count', 'auxiliary_episode_paths', 'context_episode_sha256',
              'auxiliary_windows', 'system_prompt_sha256', 'target_user_prompt_sha256', 'user_prompt_sha256')
    common.write_csv(directory / 'manifest.csv', [{k: q[k] for k in fields} for q in manifest['queries']])
    for q in manifest['queries']:
        common.write_text(directory / 'prompts' / f"{q['condition_id']}__N{q['ordinal']}.txt",
                          '[SYSTEM]\n' + q['system_prompt'] + '\n\n[USER]\n' + q['user_prompt'])
    print(f"Planned {len(manifest['queries'])} requests. Preview: {directory}")


def summarize(root):
    return icrl.summarize(root, csv_fields=CSV_FIELDS,
        condition_fields=('model_alias', 'model', 'context_method', 'H', 'E', 'context_episode_count'),
        note='Bin accuracy uses parsed bins; MAE uses parsed numeric actions. Token/time aggregates '
             'use saved scored responses. Each condition has ten distinct targets; pair_id joins '
             'fewshot4/episodes_E9 within a model. Windows overlap in episode_0.')


def run(spec, argv=None):
    parser = argparse.ArgumentParser(description='No flags: execute the 40 paired queries and summarize.')
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
            print('API requests made: 0')
            return 0
        execution = import_module('experiments.09_ablation_reward.runner.execution')
        try:
            return execution.execute(manifest, root, resume=args.resume,
                                     source_experiment='12_fewshots_episodes', csv_fields=CSV_FIELDS)
        finally:
            if (root / 'api/manifest.json').exists():
                summarize(root)
