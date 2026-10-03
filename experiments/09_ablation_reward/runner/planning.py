"""Explicit episode/start selection, paired reward prompts and repeat identities."""
import argparse
from dataclasses import asdict, dataclass
from importlib import import_module
from itertools import product
from pathlib import Path

from experiments import common
from experiments.joint_questions import JOINT_QUESTIONS
from experiments.split_execution import execution_lock
from preprocessing.llmx_original import build_prompt_queries, discover_episodes, file_sha256
from .prompts import without_rewards

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]


@dataclass(frozen=True)
class Experiment:
    models: tuple[str, ...]
    histories: tuple[int, ...]
    starts: tuple[int, ...]
    shots: tuple[int, ...]
    reward_modes: tuple[bool, ...]
    repeats: int
    episode: str


def make_plan(spec):
    if (spec.repeats < 1 or any(h < 1 for h in spec.histories)
            or any(t < 0 for t in spec.starts) or not set(spec.shots) <= {0, 4}):
        raise ValueError('Invalid experiment conditions')
    snapshot_path = HERE / 'data_prep/fewshot_p1_k4.json'
    snapshot = common.read_json(snapshot_path)
    prefix = snapshot['prefix']
    if common.digest(prefix) != snapshot['prefix_sha256']:
        raise ValueError('Frozen few-shot prefix changed')
    sources = [s for s in discover_episodes() if s.task == 'Pendulum-v1' and s.path.name == spec.episode]
    if len(sources) != 1:
        raise ValueError('Expected exactly one target episode')
    source = sources[0]
    question = JOINT_QUESTIONS['Pendulum-v1']['next-action']
    targets = {}
    for H in spec.histories:
        for q in build_prompt_queries(source, metric='next-action', question_name=question,
                                      history_size=common.history_size(H)):
            if q.history_start in spec.starts:
                if q.history_end != q.history_start + H or q.query_index != q.history_end:
                    raise ValueError('Unexpected next-action window')
                targets[H, q.history_start] = q
    if len(targets) != len(spec.histories) * len(spec.starts):
        raise ValueError('Missing target window')
    rows = []
    for alias, H, start, shots, repeat, reward in product(
            spec.models, spec.histories, spec.starts, spec.shots,
            range(1, spec.repeats + 1), spec.reward_modes):
        q = targets[H, start]
        target = q.user_prompt if reward else without_rewards(q)
        user = (prefix if shots else '') + target
        condition = f'{alias}__H{H}__t{start}__shots{shots}__reward{int(reward)}'
        row = dict(condition_id=condition, model_alias=alias, model=common.MODELS[alias],
                   task='Pendulum-v1', metric='next-action', H=H, shots=shots,
                   score_pattern='pattern1' if shots else 'none', reward_present=reward,
                   reward_scope='target_history_only', repeat_id=repeat, ordinal=repeat-1,
                   episode_path=str(source.path.relative_to(ROOT)), episode_sha256=q.dataset_sha256,
                   query_index=q.query_index, history_start=start, history_end_exclusive=start+H,
                   question_name=question, example_ids=snapshot['example_ids'] if shots else [],
                   system_prompt=q.system_prompt, user_prompt=user, target_user_prompt=target,
                   target_offset=len(prefix) if shots else 0,
                   system_prompt_sha256=common.digest(q.system_prompt), user_prompt_sha256=common.digest(user))
        row['request'] = common.expected_api_request(row['model'], q.system_prompt, user)
        row['query_id'] = common.digest(common.json_text(row))
        rows.append(row)
    if len({r['query_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate execution identities')
    return dict(experiment='09_ablation_reward', status='ready', config=asdict(spec),
                input_sha256={str(snapshot_path.relative_to(ROOT)):file_sha256(snapshot_path),
                              str(source.path.relative_to(ROOT)):file_sha256(source.path)},
                queries=rows)


def save_preview(manifest, root):
    directory = root / 'dry_run'
    directory.mkdir(parents=True, exist_ok=True)
    common.write_json(directory / 'manifest.json', manifest)
    fields = ('query_id', 'condition_id', 'model_alias', 'H', 'history_start', 'query_index',
              'shots', 'reward_present', 'repeat_id')
    common.write_csv(directory / 'manifest.csv', [{k:q[k] for k in fields} for q in manifest['queries']])
    prompts = directory / 'prompts'
    prompts.mkdir(exist_ok=True)
    for q in manifest['queries']:
        common.write_text(prompts / f"{q['condition_id']}__repeat{q['repeat_id']}.txt",
                          '[SYSTEM]\n'+q['system_prompt']+'\n\n[USER]\n'+q['user_prompt'])
    print(f"Planned {len(manifest['queries'])} requests. Preview: {directory}")


def run(spec, argv=None):
    parser = argparse.ArgumentParser(description='No flags: execute all 96 requests and summarize.')
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
        from .execution import execute
        try:
            return execute(manifest, root, resume=args.resume)
        finally:
            if (root / 'api/manifest.json').exists():
                import_module('experiments.09_ablation_reward.analysis.summarize').summarize(root)
