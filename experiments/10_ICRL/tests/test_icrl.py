from collections import Counter
from copy import deepcopy
from dataclasses import replace
from importlib import import_module
from itertools import product
from pathlib import Path
from types import SimpleNamespace
import re

import numpy as np
import pytest

entry = import_module('experiments.10_ICRL.run')
runner = entry.runner
prompts = import_module('experiments.10_ICRL.prompts')
execution = import_module('experiments.09_ablation_reward.runner.execution')
common = runner.common


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import openai
    monkeypatch.setattr(openai, 'OpenAI', lambda **kw: pytest.fail('Unexpected API client'))
    monkeypatch.setattr(execution.getpass, 'getpass', lambda prompt: pytest.fail('Unexpected key prompt'))
    monkeypatch.setattr(common, 'output_path', lambda path: Path(path))


@pytest.fixture(scope='module')
def plan():
    return runner.make_plan(entry.EXPERIMENT)


def row_for(plan, method='direct', E=1):
    return next(q for q in plan['queries'] if q['method'] == method and q['E'] == E)


def test_full_grid_and_fixed_target(plan):
    rows = plan['queries']
    assert len(rows) == len({q['query_id'] for q in rows}) == 72
    expected = set(product(('terra', 'luna'), tuple(prompts.METHOD_K), (1, 3, 9), (1, 2, 3)))
    assert {(q['model_alias'], q['method'], q['E'], q['repeat_id']) for q in rows} == expected
    assert len(Counter(q['condition_id'] for q in rows)) == 24
    assert set(Counter(q['condition_id'] for q in rows).values()) == {3}
    episode = runner.Episode.load(runner.ROOT / rows[0]['episode_path'])
    from llm_x.metrics import bin_actions
    for q in rows:
        assert (q['task'], q['metric'], q['H'], q['history_start']) == ('Pendulum-v1', 'next-action', 20, 0)
        assert q['target_episode'] == Path(q['episode_path']).name == 'episode_9_seed3407.npz'
        assert q['query_index'] == q['history_end_exclusive'] == 20
        assert q['reward_present'] is True
        assert q['action_value_ground_truth'] == episode.action_vector(20).tolist()
        assert q['ground_truth'] == bin_actions(episode.action_vector(20), start=-2, stop=2, bins=10)
        assert q['K'] == prompts.METHOD_K[q['method']]
    assert len({q['system_prompt'] for q in rows}) == 1
    for condition in {q['condition_id'] for q in rows}:
        repeats = [q for q in rows if q['condition_id'] == condition]
        assert len({common.json_text(q['request']) for q in repeats}) == 1
        assert len({q['query_id'] for q in repeats}) == 3


def test_nested_context_selection_and_hashes(plan, monkeypatch):
    sources = runner.discover_episodes()
    expected = sorted(str(s.path.relative_to(runner.ROOT)) for s in sources
                      if s.task == 'Pendulum-v1' and s.path.name != entry.EXPERIMENT.target_episode)
    contexts = {}
    from preprocessing.llmx_original import file_sha256
    for q in plan['queries']:
        assert q['E'] == q['context_episode_count']
        assert q['auxiliary_episode_paths'] == expected[:q['E']-1]
        paths = [q['episode_path'], *q['auxiliary_episode_paths']]
        assert len(set(paths)) == q['E']
        assert set(q['context_episode_sha256']) == set(paths)
        assert all(q['context_episode_sha256'][p] == file_sha256(runner.ROOT / p) for p in paths)
        contexts[q['E']] = set(paths)
    assert contexts[1] < contexts[3] < contexts[9]
    monkeypatch.setattr(runner, 'discover_episodes', lambda: list(reversed(sources)))
    assert runner.make_plan(entry.EXPERIMENT) == plan


def test_verbatim_base_and_method_differences(plan):
    source = next(s for s in runner.discover_episodes() if s.task == 'Pendulum-v1'
                  and s.path.name == entry.EXPERIMENT.target_episode)
    q = next(runner.build_prompt_queries(source, metric='next-action',
             question_name='next_action_prediction_continuous_joint', history_size=common.history_size(20)))
    assert row_for(plan)['user_prompt'] == q.user_prompt
    assert row_for(plan)['system_prompt'] == q.system_prompt
    for E in (1, 3, 9):
        direct = row_for(plan, 'direct', E)['user_prompt']
        assert direct.endswith(q.user_prompt)
        assert row_for(plan, 'ppo_prior', E)['user_prompt'] == prompts.PPO_PRIOR + '\n\n' + direct
        k2 = row_for(plan, 'icrfqi_k2', E)['user_prompt']
        k5 = row_for(plan, 'icrfqi_k5', E)['user_prompt']
        assert k2.replace('K = 2', 'K = 5') == k5
        assert k2.endswith(direct) and k5.endswith(direct)
        assert 'PPO' not in k2 and 'PPO' not in k5
        assert 'Fitted Q' not in direct and 'Fitted Q' not in row_for(plan, 'ppo_prior', E)['user_prompt']
        for phrase in ('Q_0(s, b) = 0', 'gamma = 0.99', 'all 10 action bins',
                       'y_t^(k) = r_t + gamma * max_b Q_(k-1)(s_(t+1), b)',
                       'regression/interpolation', 'without updating model parameters',
                       'b* = argmax_b Q_K(s_query, b)', 'real-valued action inside that bin',
                       'Do not print a large Q-table', 'Never create a transition across episode boundaries.'):
            assert phrase in k2


def test_auxiliary_blocks_have_s20_only_and_independent_boundaries(plan):
    from llm_x.data import _array_text
    for E in (1, 3, 9):
        q = row_for(plan, E=E)
        text = q['user_prompt']
        assert text.count('  action:') == text.count('  reward:') == 20 * E
        assert text.count('State at step 20:') == E-1
        assert 'Step 20:' not in text
        assert re.findall(r'^Step (\d+):$', text, flags=re.M) == [str(i) for i in range(20)] * E
        if E > 1:
            assert 'Never create a transition across episode boundaries.' in text
            assert text.split('[Target episode]\n', 1)[1] == q['target_user_prompt']
        for i, path in enumerate(q['auxiliary_episode_paths'], 1):
            episode = runner.Episode.load(runner.ROOT / path)
            block = text.split(f'[Additional reference episode {i}]\n', 1)[1].split(
                f'[End additional reference episode {i}]', 1)[0]
            assert block == episode.history_text(0, 20, indexed=True) + (
                f'\n\nState at step 20: {_array_text(episode.state(20))}\n')


def test_future_actions_rewards_and_states_do_not_leak(plan, monkeypatch):
    original_load = runner.Episode.load
    def changed(path):
        episode = original_load(path)
        arrays = {k: v.copy() for k, v in episode.arrays.items()}
        arrays['actions'][20:] = 0.1234567
        arrays['rewards'][20:] = -987654.321
        arrays['states'][21:] = 876543.21
        arrays['episodic_return'][...] = -765432.1
        return replace(episode, arrays=arrays)
    monkeypatch.setattr(runner.Episode, 'load', changed)
    altered = runner.make_plan(entry.EXPERIMENT)
    assert altered['queries'][0]['action_value_ground_truth'] != plan['queries'][0]['action_value_ground_truth']
    for before, after in zip(plan['queries'], altered['queries']):
        assert before['auxiliary_episode_paths'] == after['auxiliary_episode_paths']
        assert before['request'] == after['request']


def test_auxiliary_s20_reaches_prompt(plan, monkeypatch):
    original_load = runner.Episode.load
    path = runner.ROOT / row_for(plan, E=3)['auxiliary_episode_paths'][0]
    def changed(p):
        episode = original_load(p)
        if Path(p) == path:
            arrays = {k: v.copy() for k, v in episode.arrays.items()}
            arrays['states'][20] = np.array([[0.1234, 0.5678, 1.2345]])
            return replace(episode, arrays=arrays)
        return episode
    monkeypatch.setattr(runner.Episode, 'load', changed)
    altered = runner.make_plan(entry.EXPERIMENT)
    assert row_for(altered)['request'] == row_for(plan)['request']
    assert row_for(altered, E=3)['request'] != row_for(plan, E=3)['request']
    assert 'State at step 20: [[0.1234, 0.5678, 1.2345]]' in row_for(altered, E=9)['user_prompt']


def test_request_contract(plan):
    for q in plan['queries']:
        assert q['model'] == {'terra': 'gpt-5.6-terra', 'luna': 'gpt-5.6-luna'}[q['model_alias']]
        assert q['request'] == common.expected_api_request(q['model'], q['system_prompt'], q['user_prompt'])
        assert q['request'] == {'model': q['model'], 'reasoning_effort': 'medium', 'messages': [
            {'role': 'system', 'content': q['system_prompt']}, {'role': 'user', 'content': q['user_prompt']}]}


def test_dry_run_no_client_or_request(plan, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'HERE', tmp_path)
    assert runner.run(entry.EXPERIMENT, ['--dry-run']) == 0
    directory = tmp_path / 'results/dry_run'
    assert len(common.read_json(directory / 'manifest.json')['queries']) == 72
    assert len(list((directory / 'prompts').glob('*.txt'))) == 72
    for q in plan['queries']:
        text = (directory / 'prompts' / f"{q['condition_id']}__repeat{q['repeat_id']}.txt").read_text()
        assert text == '[SYSTEM]\n' + q['system_prompt'] + '\n\n[USER]\n' + q['user_prompt']
    assert not (tmp_path / 'results/api').exists()
    assert not list(tmp_path.rglob('requests.jsonl'))
    assert runner.run(entry.EXPERIMENT, ['--dry-run']) == 0


def mock_client(monkeypatch, *, fail_at=None):
    calls = []
    class Client:
        def __init__(self, **kwargs):
            assert kwargs == dict(api_key='offline-test', max_retries=0, timeout=180.0)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == fail_at:
                raise RuntimeError('simulated failure')
            raw = dict(choices=[{'message': {'content': 'predictions = [1.91]\n>>Final action bins: [9]'}}],
                       usage=dict(prompt_tokens=10, completion_tokens=5, total_tokens=15))
            return SimpleNamespace(model_dump=lambda **kw: raw)
        def close(self):
            pass
    import openai
    monkeypatch.setattr(openai, 'OpenAI', Client)
    monkeypatch.setattr(execution.getpass, 'getpass', lambda prompt: 'offline-test')
    return calls


def small_run(plan, tmp_path, monkeypatch):
    manifest = deepcopy(plan)
    manifest['queries'] = manifest['queries'][:3]
    monkeypatch.setattr(runner, 'HERE', tmp_path)
    monkeypatch.setattr(runner, 'make_plan', lambda spec: manifest)
    return manifest


def test_execution_scoring_summary_and_resume(plan, tmp_path, monkeypatch):
    manifest = small_run(plan, tmp_path, monkeypatch)
    calls = mock_client(monkeypatch)
    assert runner.run(entry.EXPERIMENT, []) == 0
    root = tmp_path / 'results'
    records = common.read_jsonl(root / 'api/records.jsonl')
    assert len(calls) == len(records) == 3
    for q, r in zip(manifest['queries'], records):
        expected = runner.scoring.score(q, r['assistant_text'], runner.Episode.load(runner.ROOT / q['episode_path']))
        assert all(r[k] == v for k, v in expected.items())
        assert r['source_experiment'] == '10_ICRL'
        assert r['status'] == 'match' and r['action_value_prediction'] == [1.91]
        assert r['raw_response']['usage']['total_tokens'] == r['total_tokens'] == 15
        assert r['request_elapsed_seconds'] >= 0
    metric = common.read_json(root / 'analysis/summary.json')['conditions'][0]
    assert metric['parse_successes'] == metric['parsed_bins'] == metric['parsed_actions'] == 3
    assert metric['bin_accuracy'] == 1
    assert metric['action_mae'] == pytest.approx(abs(1.91 - records[0]['action_value_ground_truth'][0]))
    assert metric['input_tokens_sum'] == 30 and metric['total_tokens_sum'] == 45
    assert 'parse_success' in (root / 'analysis/per_query.csv').read_text()
    with pytest.raises(FileExistsError):
        runner.run(entry.EXPERIMENT, [])
    monkeypatch.setattr(execution.getpass, 'getpass', lambda prompt: pytest.fail('Completed run must not prompt'))
    assert runner.run(entry.EXPERIMENT, ['--resume']) == 0
    assert len(calls) == 3
    # Recover saved responses after an interrupted scoring step without another request.
    (root / 'api/records.jsonl').write_text('')
    assert runner.run(entry.EXPERIMENT, ['--resume']) == 0
    assert len(calls) == 3 and len(common.read_jsonl(root / 'api/records.jsonl')) == 3


def test_failed_request_never_resent(plan, tmp_path, monkeypatch):
    small_run(plan, tmp_path, monkeypatch)
    calls = mock_client(monkeypatch, fail_at=2)
    with pytest.raises(RuntimeError, match='simulated failure'):
        runner.run(entry.EXPERIMENT, [])
    assert len(calls) == 2
    assert runner.run(entry.EXPERIMENT, ['--resume']) == 2
    assert len(calls) == 3
    metrics = common.read_json(tmp_path / 'results/analysis/summary.json')['conditions'][0]
    assert metrics['responses'] == 2 and metrics['unresolved'] == 1 and metrics['unattempted'] == 0


def test_changed_preview_is_not_overwritten(plan, tmp_path):
    runner.save_preview(plan, tmp_path)
    previous = (tmp_path / 'dry_run/manifest.json').read_bytes()
    altered = deepcopy(plan)
    altered['queries'][0]['user_prompt'] += 'changed'
    with pytest.raises(FileExistsError, match='Existing dry run differs'):
        runner.save_preview(altered, tmp_path)
    assert (tmp_path / 'dry_run/manifest.json').read_bytes() == previous


def test_invalid_or_insufficient_input(monkeypatch):
    with pytest.raises(ValueError, match='H20'):
        runner.make_plan(replace(entry.EXPERIMENT, H=5))
    sources = runner.discover_episodes()
    monkeypatch.setattr(runner, 'discover_episodes', lambda: [s for s in sources if s.path.name == entry.EXPERIMENT.target_episode])
    with pytest.raises(ValueError, match='auxiliary episodes'):
        runner.make_plan(entry.EXPERIMENT)


@pytest.mark.parametrize('text,bin_ok,action_ok', [
    ('predictions = [1.91]\n>>Final action bins: [9]', True, True),
    ('predictions = [1.91]', False, True),
    ('>>Final action bins: [9]', True, False),
    ('invalid', False, False),
    ('', False, False),
])
def test_parse_success_denominators(plan, tmp_path, text, bin_ok, action_ok):
    q = row_for(plan)
    common.write_json(tmp_path / 'api/manifest.json', {**plan, 'queries': [q]})
    evaluation = runner.scoring.score(q, text, runner.Episode.load(runner.ROOT / q['episode_path']))
    common.append_jsonl(tmp_path / 'api/requests.jsonl', {'query_id': q['query_id']})
    common.append_jsonl(tmp_path / 'api/records.jsonl', {**q, **evaluation, 'assistant_text': text})
    metric = runner.summarize(tmp_path)[0]
    assert metric['parsed_bins'] == int(bin_ok)
    assert metric['parsed_actions'] == int(action_ok)
    assert metric['parse_successes'] == int(bin_ok and action_ok)
    assert (metric['bin_accuracy'] is not None) == bin_ok
    assert (metric['action_mae'] is not None) == action_ok
