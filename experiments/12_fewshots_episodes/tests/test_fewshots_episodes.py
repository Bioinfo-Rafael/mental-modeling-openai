from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
from importlib import import_module
from itertools import islice, product
from pathlib import Path
import csv
import re

import pytest

entry = import_module('experiments.12_fewshots_episodes.run')
runner = entry.runner
common = runner.common
execution = import_module('experiments.09_ablation_reward.runner.execution')
fewshot = import_module('experiments.08_fewshot.runner.fewshot_runner')
icrl_tests = import_module('experiments.10_ICRL.tests.test_icrl')


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import httpx
    import openai
    monkeypatch.setattr(openai, 'OpenAI', lambda **kw: pytest.fail('Unexpected API client'))
    monkeypatch.setattr(httpx.Client, 'send', lambda *a, **kw: pytest.fail('Network forbidden'))
    monkeypatch.setattr(execution.getpass, 'getpass', lambda prompt: pytest.fail('Unexpected key prompt'))
    monkeypatch.setattr(common, 'output_path', lambda path: Path(path))


@pytest.fixture(scope='module')
def plan():
    return runner.make_plan(entry.EXPERIMENT)


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def test_grid_and_windows(plan):
    rows = plan['queries']
    assert len(rows) == len({q['query_id'] for q in rows}) == 40
    assert {(q['model_alias'], q['context_method'], q['ordinal']) for q in rows} == set(
        product(('terra', 'luna'), ('fewshot4', 'episodes_E9'), range(10)))
    assert Counter(q['condition_id'] for q in rows) == {
        f'{m}__{c}': 10 for m, c in product(('terra', 'luna'), ('fewshot4', 'episodes_E9'))}
    for q in rows:
        assert (q['task'], q['metric'], q['H']) == ('Pendulum-v1', 'next-action', 20)
        assert q['target_episode'] == Path(q['episode_path']).name == 'episode_0.npz'
        assert q['history_start'] == q['ordinal']
        assert q['history_end_exclusive'] == q['query_index'] == 20 + q['ordinal']
        assert q['question_name'] == 'next_action_prediction_continuous_joint'
        assert 'repeat_id' not in q


def test_saved_05_06_03_1_identity_and_prompt_hashes(plan):
    root = common.EXPERIMENTS
    merged = sorted((root / '06_new_models_n10/results/merged').glob('*/manifest.csv'))
    assert merged, 'Expected saved Exp.06 merged manifests'
    paths = [root / '05_new_models_single/results/manifest.csv',
             root / '03_1_gpt35_history_n30_joint/results/manifest.csv', *merged]
    for path in paths:
        selected = [q for q in read_csv(path) if q['task'] == 'Pendulum-v1'
                    and q['metric'] == 'next-action' and int(q['H']) == 20
                    and int(q['ordinal']) < 10 and q['model_alias'] in ('terra', 'luna', '3.5')]
        aliases = {'3.5'} if '03_1_gpt35_history_n30_joint' in path.parts else {'terra', 'luna'}
        assert Counter(q['model_alias'] for q in selected) == {alias: 10 for alias in aliases}
        for old in selected:
            paired = [q for q in plan['queries'] if q['ordinal'] == int(old['ordinal'])
                      and q['model_alias'] == ('terra' if old['model_alias'] == '3.5' else old['model_alias'])]
            assert len(paired) == 2
            for new in paired:
                assert old['episode'] == new['target_episode']
                for field in ('episode_path', 'episode_sha256', 'question_name', 'system_prompt_sha256'):
                    assert old[field] == new[field]
                for field in ('history_start', 'query_index'):
                    assert int(old[field]) == new[field]
                assert int(old['history_end']) == new['history_end_exclusive']
                assert old['user_prompt_sha256'] == new['target_user_prompt_sha256']
                assert old['user_prompt_sha256'] == common.digest(new['target_user_prompt'])


def test_frozen_examples_verbatim_order_and_no_direct_target_labels(plan):
    snapshot = common.read_json(runner.SNAPSHOT)
    source = common.read_json(runner.ROOT / snapshot['source_manifest'])
    old = next(q for q in source['queries'] if q['query_id'] == snapshot['source_query_id'])
    assert snapshot['prefix'] == old['user_prompt'][:old['target_offset']]
    assert snapshot['example_ids'] == old['example_ids']
    assert common.digest(snapshot['prefix']) == snapshot['prefix_sha256']
    candidates = common.read_json(common.EXPERIMENTS / '08_fewshot/results/dry_run/selected_candidates.json')['examples']
    examples = [next(e for e in candidates if e['query_id'] == qid) for qid in snapshot['example_ids']]
    target = plan['queries'][0]['target_user_prompt']
    rebuilt, offset = fewshot.compose_user(examples, target)  # Audit only; no selection/rendering at runtime.
    assert rebuilt[:offset] == snapshot['prefix']
    forbidden = set(range(20, 30))
    for e in examples:
        assert Path(e['episode_path']).name == 'episode_0.npz'
        assert e['H'] == 5 and e['query_index'] not in forbidden
        steps = {int(s) for s in re.findall(r'^Step (\d+):$', e['question'], re.M)}
        assert len(steps) == e['H'] and not steps & forbidden
        assert e['example_type'] in ('correct', 'incorrect')
        assert e['reasoning_score'] in (1, 3)
        # Question/Answer refer only to earlier actions; Label belongs to that query_index.
        for text in (e['question'], e['answer']):
            assert not re.search(r'\ba_?\{?(?:2[0-9])\b|\bstep\s+2[0-9]\b', text, re.I)
    assert [(e['metric'], e['query_index']) for e in examples] == [
        ('last-action', 10), ('next-action', 8), ('last-action', 10), ('next-action', 13)]
    for q in plan['queries']:
        if q['context_method'] == 'fewshot4':
            assert (q['shots'], q['score_pattern']) == (4, 'pattern1')
            assert q['example_ids'] == snapshot['example_ids']
            assert q['fewshot_prefix_sha256'] == snapshot['prefix_sha256']
            assert q['fewshot_source'] == str(runner.SNAPSHOT.relative_to(runner.ROOT))
            assert q['user_prompt'] == snapshot['prefix'] + q['target_user_prompt']
            assert q['user_prompt'].count('New Question:') == 1


def test_paired_target_and_only_prefix_changes(plan):
    source = next(s for s in runner.discover_episodes() if s.path == runner.ROOT / plan['queries'][0]['episode_path'])
    targets = list(islice(runner.build_prompt_queries(source, metric='next-action',
        question_name='next_action_prediction_continuous_joint', history_size=common.history_size(20)), 10))
    pairs = defaultdict(list)
    for q in plan['queries']:
        pairs[q['pair_id']].append(q)
        original = targets[q['ordinal']]
        assert q['system_prompt'] == original.system_prompt
        assert q['target_user_prompt'] == original.user_prompt
        assert q['user_prompt'][q['target_offset']:] == original.user_prompt
        assert q['user_prompt'].endswith(original.question_text.strip())
        assert runner.prompts.PPO_PRIOR not in q['user_prompt']
        assert 'Fitted Q-Iteration' not in q['user_prompt']
        assert 'Q_0(s, b)' not in q['user_prompt']
        assert '>>Final action bins:' in original.user_prompt
    assert len(pairs) == 20
    for rows in pairs.values():
        assert len(rows) == 2 and {q['context_method'] for q in rows} == {'fewshot4', 'episodes_E9'}
        a, b = rows
        for key in ('model', 'ordinal', 'episode_path', 'episode_sha256', 'history_start',
                    'history_end_exclusive', 'query_index', 'ground_truth', 'action_value_ground_truth',
                    'system_prompt', 'target_user_prompt', 'question_name'):
            assert a[key] == b[key]
        assert a['user_prompt'][:a['target_offset']] != b['user_prompt'][:b['target_offset']]
    for ordinal, context in product(range(10), ('fewshot4', 'episodes_E9')):
        rows = [q for q in plan['queries'] if q['ordinal'] == ordinal and q['context_method'] == context]
        assert rows[0]['user_prompt'] == rows[1]['user_prompt']


def test_e9_selection_hashes_windows_and_boundaries(plan):
    from llm_x.data import _array_text
    expected = sorted(str(s.path.relative_to(runner.ROOT)) for s in runner.discover_episodes()
                      if s.task == 'Pendulum-v1' and s.path.name != 'episode_0.npz')[:8]
    for q in plan['queries']:
        if q['context_method'] != 'episodes_E9':
            continue
        assert q['E'] == q['context_episode_count'] == 9
        assert q['auxiliary_episode_paths'] == expected
        assert len(set([q['episode_path'], *expected])) == 9
        assert set(q['context_episode_sha256']) == {q['episode_path'], *expected}
        assert all(q['context_episode_sha256'][p] == runner.file_sha256(runner.ROOT / p)
                   for p in q['context_episode_sha256'])
        episodes = [runner.Episode.load(runner.ROOT / p) for p in expected]
        prefix = q['user_prompt'][:q['target_offset']]
        assert prefix == runner.prompts.reference_prefix(episodes, start=q['history_start'], H=20)
        assert q['user_prompt'] == runner.prompts.compose_user(q['target_user_prompt'], prefix, 'direct')
        assert 'Each episode is an independent trajectory.' in prefix
        assert 'Never create a transition across episode boundaries.' in prefix
        blocks = re.findall(r'\[Additional reference episode (\d+)\]\n(.*?)\n\[End additional reference episode \1\]', prefix, re.S)
        assert len(blocks) == 8
        start, end = q['history_start'], q['query_index']
        for i, ((number, block), episode, window) in enumerate(zip(blocks, episodes, q['auxiliary_windows']), 1):
            assert int(number) == i
            assert window == dict(episode_path=expected[i-1], history_start=start,
                                 history_end_exclusive=end, final_state_index=end)
            assert re.findall(r'^Step (\d+):$', block, re.M) == [str(t) for t in range(start, end)]
            assert block.count('  action:') == block.count('  reward:') == 20
            assert block == episode.history_text(start, end, indexed=True) + f'\n\nState at step {end}: {_array_text(episode.state(end))}'
            assert f'Step {end}:' not in block
        assert prefix.endswith('[Target episode]\n')


def test_selection_deterministic(plan, monkeypatch):
    sources = runner.discover_episodes()
    monkeypatch.setattr(runner, 'discover_episodes', lambda: list(reversed(sources)))
    assert runner.make_plan(entry.EXPERIMENT) == plan


@pytest.mark.parametrize('ordinal', range(10))
def test_query_actions_rewards_and_future_do_not_enter_request(plan, monkeypatch, ordinal):
    original_load = runner.Episode.load
    index = ordinal + 20
    def changed(path):
        episode = original_load(path)
        arrays = {k: v.copy() for k, v in episode.arrays.items()}
        arrays['actions'][index:] = 0.1234567
        arrays['rewards'][index:] = -987654.321
        arrays['states'][index + 1:] = 876543.21
        arrays['episodic_return'][...] = -765432.1
        return replace(episode, arrays=arrays)
    monkeypatch.setattr(runner.Episode, 'load', changed)
    altered = runner.make_plan(entry.EXPERIMENT)
    for before, after in zip(plan['queries'], altered['queries']):
        if before['ordinal'] == ordinal:
            assert before['action_value_ground_truth'] != after['action_value_ground_truth']
            assert before['request'] == after['request']


def test_api_parameters_match_existing_terra_luna(plan):
    old = common.read_json(common.EXPERIMENTS / '05_new_models_single/results/manifest.json')
    for q in plan['queries']:
        assert q['model'] == {'terra': 'gpt-5.6-terra', 'luna': 'gpt-5.6-luna'}[q['model_alias']]
        assert q['request'] == common.expected_api_request(q['model'], q['system_prompt'], q['user_prompt'])
        assert {k: v for k, v in q['request'].items() if k not in ('messages', 'model')} == old[
            'api_request_options_by_model'][q['model']] == {'reasoning_effort': 'medium'}


def test_dry_run_zero_client_and_protect_existing_preview(plan, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'HERE', tmp_path)
    monkeypatch.setattr(execution, 'execute', lambda *a, **kw: pytest.fail('Dry run reached executor'))
    assert runner.run(entry.EXPERIMENT, ['--dry-run']) == 0
    directory = tmp_path / 'results/dry_run'
    assert common.read_json(directory / 'manifest.json')['queries'] == plan['queries']
    assert len(read_csv(directory / 'manifest.csv')) == 40
    assert len(list((directory / 'prompts').glob('*.txt'))) == 40
    for q in plan['queries']:
        text = (directory / 'prompts' / f"{q['condition_id']}__N{q['ordinal']}.txt").read_text()
        assert text == '[SYSTEM]\n' + q['system_prompt'] + '\n\n[USER]\n' + q['user_prompt']
    assert not (tmp_path / 'results/api').exists()
    assert not list(tmp_path.rglob('requests.jsonl'))
    assert runner.run(entry.EXPERIMENT, ['--dry-run']) == 0
    changed = deepcopy(plan)
    changed['queries'][0]['user_prompt'] += 'changed'
    with pytest.raises(FileExistsError, match='Existing dry run differs'):
        runner.save_preview(changed, tmp_path / 'results')
    assert common.read_json(directory / 'manifest.json')['queries'] == plan['queries']


def test_mock_execution_scorer_summary_and_resume(plan, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'HERE', tmp_path)
    calls = icrl_tests.mock_client(monkeypatch)
    assert runner.run(entry.EXPERIMENT, []) == 0
    root = tmp_path / 'results'
    records = common.read_jsonl(root / 'api/records.jsonl')
    assert len(calls) == len(records) == 40
    assert runner.scoring is runner.icrl.scoring is execution.scoring
    episode = runner.Episode.load(runner.ROOT / records[0]['episode_path'])
    for q, r in zip(plan['queries'], records):
        assert all(r[k] == v for k, v in runner.scoring.score(q, r['assistant_text'], episode).items())
        assert r['source_experiment'] == '12_fewshots_episodes'
        assert r['raw_response']['usage']['total_tokens'] == r['total_tokens'] == 15
        assert r['request_elapsed_seconds'] >= 0
    metrics = common.read_json(root / 'analysis/summary.json')['conditions']
    assert len(metrics) == 4
    for metric in metrics:
        rows = [r for r in records if r['condition_id'] == metric['condition_id']]
        assert metric['planned'] == metric['responses'] == metric['parse_successes'] == 10
        assert metric['context_method'] == rows[0]['context_method']
        assert metric['bin_accuracy'] == metric['match_rate_all_planned'] == sum(r['status'] == 'match' for r in rows) / 10
        assert metric['action_mae'] == pytest.approx(sum(r['absolute_error'] for r in rows) / 10)
        for name, value in (('input_tokens', 10), ('output_tokens', 5), ('total_tokens', 15)):
            assert metric[name + '_mean'] == value and metric[name + '_sum'] == 10 * value
        assert metric['request_elapsed_seconds_mean'] >= 0
    per_query = read_csv(root / 'analysis/per_query.csv')
    assert len(per_query) == 40
    assert Counter(r['pair_id'] for r in per_query) == Counter({q['pair_id']: 2 for q in plan['queries']})
    assert all(r['parse_success'] == 'True' and r['raw_response'] for r in per_query)
    with pytest.raises(FileExistsError):
        runner.run(entry.EXPERIMENT, [])
    monkeypatch.setattr(execution.getpass, 'getpass', lambda prompt: pytest.fail('Completed run must not prompt'))
    assert runner.run(entry.EXPERIMENT, ['--resume']) == 0
    (root / 'api/records.jsonl').write_text('')
    assert runner.run(entry.EXPERIMENT, ['--resume']) == 0
    assert len(calls) == 40


def test_failed_mock_request_never_resent(plan, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'HERE', tmp_path)
    calls = icrl_tests.mock_client(monkeypatch, fail_at=2)
    with pytest.raises(RuntimeError, match='simulated failure'):
        runner.run(entry.EXPERIMENT, [])
    assert runner.run(entry.EXPERIMENT, ['--resume']) == 2
    assert len(calls) == 40
    metrics = common.read_json(tmp_path / 'results/analysis/summary.json')['conditions']
    assert sum(m['unresolved'] for m in metrics) == 1
    assert sum(m['responses'] for m in metrics) == 39


@pytest.mark.parametrize('text,bin_ok,action_ok', [
    ('predictions = [1.91]\n>>Final action bins: [9]', True, True),
    ('predictions = [1.91]', False, True), ('>>Final action bins: [9]', True, False),
    ('invalid', False, False), ('', False, False),
])
def test_parse_denominators_both_contexts(plan, tmp_path, text, bin_ok, action_ok):
    queries = [q for q in plan['queries'] if q['ordinal'] == 0 and q['model_alias'] == 'terra']
    common.write_json(tmp_path / 'api/manifest.json', {**plan, 'queries': queries})
    for q in queries:
        evaluation = runner.scoring.score(q, text, runner.Episode.load(runner.ROOT / q['episode_path']))
        common.append_jsonl(tmp_path / 'api/requests.jsonl', {'query_id': q['query_id']})
        common.append_jsonl(tmp_path / 'api/records.jsonl', {**q, **evaluation, 'assistant_text': text})
    metrics = runner.summarize(tmp_path)
    assert len(metrics) == 2
    for m in metrics:
        assert m['parsed_bins'] == int(bin_ok) and m['parsed_actions'] == int(action_ok)
        assert m['parse_successes'] == int(bin_ok and action_ok)
        assert (m['bin_accuracy'] is not None) == bin_ok
        assert (m['action_mae'] is not None) == action_ok


def test_invalid_spec_and_inputs(monkeypatch, tmp_path):
    for changes in (dict(H=5), dict(n=1), dict(task='MountainCar-v0'), dict(metric='last-action'),
                    dict(models=('terra', 'terra')), dict(context_methods=('fewshot4', 'fewshot4'))):
        with pytest.raises(ValueError):
            runner.make_plan(replace(entry.EXPERIMENT, **changes))
    snapshot = common.read_json(runner.SNAPSHOT)
    snapshot['prefix'] += 'changed'
    path = tmp_path / 'snapshot.json'
    common.write_json(path, snapshot)
    with monkeypatch.context() as patch:
        patch.setattr(runner, 'SNAPSHOT', path)
        with pytest.raises(ValueError, match='prefix changed'):
            runner.make_plan(entry.EXPERIMENT)
    sources = runner.discover_episodes()
    monkeypatch.setattr(runner, 'discover_episodes', lambda: [s for s in sources if s.path.name == 'episode_0.npz'])
    with pytest.raises(ValueError, match='eight auxiliary'):
        runner.make_plan(entry.EXPERIMENT)
    monkeypatch.setattr(runner, 'discover_episodes', lambda: [])
    with pytest.raises(ValueError, match='episode_0'):
        runner.make_plan(entry.EXPERIMENT)
