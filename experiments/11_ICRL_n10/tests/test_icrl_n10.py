from collections import Counter
from copy import deepcopy
from dataclasses import replace
from importlib import import_module
from itertools import islice, product
from pathlib import Path
import csv
import json
import re

import pytest

entry = import_module('experiments.11_ICRL_n10.run')
runner = entry.runner
common = runner.common
execution = import_module('experiments.09_ablation_reward.runner.execution')
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


def test_grid_and_target_identity(plan):
    rows = plan['queries']
    assert len(rows) == len({q['query_id'] for q in rows}) == 40
    assert {(q['model_alias'], q['reward_present'], q['ordinal']) for q in rows} == set(
        product(('terra', 'luna'), (True, False), range(10)))
    assert len(Counter(q['condition_id'] for q in rows)) == 4
    assert set(Counter(q['condition_id'] for q in rows).values()) == {10}
    for q in rows:
        assert (q['task'], q['metric'], q['H'], q['method'], q['K'], q['E']) == (
            'Pendulum-v1', 'next-action', 20, 'icrfqi_k5', 5, 1)
        assert q['target_episode'] == Path(q['episode_path']).name == 'episode_0.npz'
        assert q['history_start'] == q['ordinal']
        assert q['history_end_exclusive'] == q['query_index'] == 20 + q['ordinal']
        assert q['context_episode_count'] == 1 and q['auxiliary_episode_paths'] == []
        assert q['question_name'] == 'next_action_prediction_continuous_joint'
        assert q['reward_mode'] == ('with_reward' if q['reward_present'] else 'without_reward')
        assert q['context_episode_sha256'] == plan['input_sha256']


def test_matches_saved_05_06_03_1_manifests(plan):
    root = common.EXPERIMENTS
    merged = sorted((root / '06_new_models_n10/results/merged').glob('*/manifest.csv'))
    assert merged, 'Expected the existing Exp.06 merged manifest'
    paths = [root / '05_new_models_single/results/manifest.csv',
             root / '03_1_gpt35_history_n30_joint/results/manifest.csv', *merged]
    for path in paths:
        selected = [q for q in read_csv(path) if q['task'] == 'Pendulum-v1'
                    and q['metric'] == 'next-action' and int(q['H']) == 20
                    and int(q['ordinal']) < 10 and q['model_alias'] in ('terra', 'luna', '3.5')]
        aliases = {'3.5'} if '03_1_gpt35_history_n30_joint' in path.parts else {'terra', 'luna'}
        assert Counter(q['model_alias'] for q in selected) == {alias: 10 for alias in aliases}
        for old in selected:
            new = next(q for q in plan['queries'] if q['ordinal'] == int(old['ordinal'])
                       and q['model_alias'] == ('terra' if old['model_alias'] == '3.5' else old['model_alias'])
                       and q['reward_present'])
            assert old['episode'] == new['target_episode']
            for field in ('episode_path', 'episode_sha256', 'question_name', 'system_prompt_sha256'):
                assert old[field] == new[field]
            for field in ('history_start', 'query_index'):
                assert int(old[field]) == new[field]
            assert int(old['history_end']) == new['history_end_exclusive']
            assert old['user_prompt_sha256'] == new['base_target_user_prompt_sha256']
            assert old['user_prompt_sha256'] == common.digest(new['target_user_prompt'])
            if '06_new_models_n10' in path.parts:
                assert old['reuse_required'] == 'True'


def test_exact_prompt_pairing_and_upstream_reuse(plan):
    source = next(s for s in runner.discover_episodes() if s.path == runner.ROOT / plan['queries'][0]['episode_path'])
    targets = list(islice(runner.build_prompt_queries(source, metric='next-action',
        question_name='next_action_prediction_continuous_joint', history_size=19), 10))
    pairs = {}
    for q in plan['queries']:
        pairs.setdefault(q['pair_id'], {})[q['reward_present']] = q
    assert len(pairs) == 20
    instruction = runner.prompts.ICRFQI.replace('{K}', '5') + '\n\n'
    for pair in pairs.values():
        assert set(pair) == {True, False}
        on, off = pair[True], pair[False]
        original = targets[on['ordinal']]
        for key in runner.IDENTITY_FIELDS:
            if key not in ('reward_present', 'reward_mode'):
                assert on[key] == off[key]
        assert on['ground_truth'] == off['ground_truth']
        assert on['action_value_ground_truth'] == off['action_value_ground_truth']
        assert on['system_prompt'] == off['system_prompt'] == original.system_prompt
        assert 'Reward space:' in off['system_prompt']
        assert on['target_user_prompt'] == original.user_prompt
        assert off['target_user_prompt'] == runner.without_rewards(original)
        assert on['user_prompt'] == runner.prompts.compose_user(original.user_prompt, '', 'icrfqi_k5')
        assert on['user_prompt'] == instruction + original.user_prompt
        assert off['user_prompt'] == instruction + off['target_user_prompt']
        rewards = [line for line in on['user_prompt'].splitlines() if line.startswith('  reward:')]
        assert len(rewards) == 20
        for line in rewards:
            value = json.loads(line.split(':', 1)[1])
            assert len(value) == 1 and isinstance(value[0], (int, float))
        assert '  reward:' not in off['user_prompt']
        expected = ''.join(line for line in on['user_prompt'].splitlines(keepends=True)
                           if not line.startswith('  reward:'))
        assert off['user_prompt'] == expected
        for q in (on, off):
            assert q['user_prompt'].endswith(original.question_text.strip())
            assert re.findall(r'^Step (\d+):$', q['user_prompt'], re.M) == [
                str(i) for i in range(q['ordinal'], q['ordinal'] + 20)]
            assert q['user_prompt'].count('  state:') == q['user_prompt'].count('  action:') == 20


@pytest.mark.parametrize('ordinal', range(10))
def test_target_action_reward_and_future_do_not_leak(plan, monkeypatch, ordinal):
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


def test_request_contract(plan):
    old = common.read_json(common.EXPERIMENTS / '05_new_models_single/results/manifest.json')
    for q in plan['queries']:
        options = old['api_request_options_by_model'][q['model']]
        assert q['request'] == common.expected_api_request(q['model'], q['system_prompt'], q['user_prompt'])
        assert q['model'] == {'terra': 'gpt-5.6-terra', 'luna': 'gpt-5.6-luna'}[q['model_alias']]
        assert {k: v for k, v in q['request'].items() if k not in ('messages', 'model')} == options == {
            'reasoning_effort': 'medium'}


def test_dry_run_no_client_or_request(plan, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'HERE', tmp_path)
    monkeypatch.setattr(execution, 'execute', lambda *a, **kw: pytest.fail('Dry run reached executor'))
    assert runner.run(entry.EXPERIMENT, ['--dry-run']) == 0
    directory = tmp_path / 'results/dry_run'
    assert common.read_json(directory / 'manifest.json')['queries'] == plan['queries']
    assert len(read_csv(directory / 'manifest.csv')) == 40
    assert len(list((directory / 'prompts').glob('*.txt'))) == 40
    for q in plan['queries']:
        text = (directory / 'prompts' / f"{q['condition_id']}__query{q['ordinal']:02d}.txt").read_text()
        assert text == '[SYSTEM]\n' + q['system_prompt'] + '\n\n[USER]\n' + q['user_prompt']
    assert not (tmp_path / 'results/api').exists()
    assert not list(tmp_path.rglob('requests.jsonl'))
    assert runner.run(entry.EXPERIMENT, ['--dry-run']) == 0
    changed = deepcopy(plan)
    changed['queries'][0]['user_prompt'] += 'changed'
    with pytest.raises(FileExistsError, match='Existing dry run differs'):
        runner.save_preview(changed, tmp_path / 'results')
    assert common.read_json(directory / 'manifest.json')['queries'] == plan['queries']


def test_mock_execution_scoring_four_summaries_and_resume(plan, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'HERE', tmp_path)
    calls = icrl_tests.mock_client(monkeypatch)
    assert runner.run(entry.EXPERIMENT, []) == 0
    root = tmp_path / 'results'
    records = common.read_jsonl(root / 'api/records.jsonl')
    assert len(calls) == len(records) == 40
    assert runner.scoring is runner.icrl.scoring
    episode = runner.Episode.load(runner.ROOT / records[0]['episode_path'])
    for q, r in zip(plan['queries'], records):
        expected = runner.scoring.score(q, r['assistant_text'], episode)
        assert all(r[k] == v for k, v in expected.items())
        assert r['source_experiment'] == '11_ICRL_n10'
        assert r['input_tokens'] == 10 and r['total_tokens'] == 15
        assert r['request_elapsed_seconds'] >= 0
    metrics = common.read_json(root / 'analysis/summary.json')['conditions']
    assert len(metrics) == 4
    for metric in metrics:
        rows = [r for r in records if r['condition_id'] == metric['condition_id']]
        assert metric['planned'] == metric['responses'] == metric['parse_successes'] == 10
        assert metric['bin_accuracy'] == sum(r['status'] == 'match' for r in rows) / 10
        assert metric['action_mae'] == pytest.approx(sum(r['absolute_error'] for r in rows) / 10)
        assert metric['reward_present'] == rows[0]['reward_present']
        assert metric['reward_mode'] == rows[0]['reward_mode']
        assert metric['input_tokens_sum'] == 100
    per_query = read_csv(root / 'analysis/per_query.csv')
    assert len(per_query) == 40
    assert Counter(r['pair_id'] for r in per_query) == Counter({q['pair_id']: 2 for q in plan['queries']})
    for row in per_query:
        assert row['parse_success'] == 'True'
        assert all(row[key] for key in runner.IDENTITY_FIELDS)
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
    assert len(calls) == 2
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
def test_parse_denominators_both_conditions(plan, tmp_path, text, bin_ok, action_ok):
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


def test_invalid_spec_and_missing_target(monkeypatch):
    for changes in (dict(H=5), dict(n=1), dict(method='direct'), dict(models=('terra', 'terra')),
                    dict(reward_modes=(True, True)), dict(reward_modes=(1, 0))):
        with pytest.raises(ValueError):
            runner.make_plan(replace(entry.EXPERIMENT, **changes))
    monkeypatch.setattr(runner, 'discover_episodes', lambda: [])
    with pytest.raises(ValueError, match='episode_0'):
        runner.make_plan(entry.EXPERIMENT)
