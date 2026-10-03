from copy import deepcopy
from importlib import import_module
from types import SimpleNamespace
from collections import Counter
from pathlib import Path

import pytest

entry = import_module('experiments.09_ablation_reward.run')
planning = import_module('experiments.09_ablation_reward.runner.planning')
execution = import_module('experiments.09_ablation_reward.runner.execution')
analysis = import_module('experiments.09_ablation_reward.analysis.summarize')
common = planning.common


@pytest.fixture(autouse=True)
def isolated_test_outputs(monkeypatch):
    # The production writer restricts output to experiments/; tests use pytest temp dirs.
    monkeypatch.setattr(common, 'output_path', lambda path: Path(path))


@pytest.fixture(scope='module')
def plan():
    return planning.make_plan(entry.EXPERIMENT)


def test_grid_windows_repeats_and_reward_pairing(plan):
    rows = plan['queries']
    assert len(rows) == len({r['query_id'] for r in rows}) == 96
    assert set(Counter(r['condition_id'] for r in rows).values()) == {3}
    assert len({r['condition_id'] for r in rows}) == 32
    assert {r['query_index'] for r in rows} == {5, 20, 105, 120}
    assert all(r['episode_path'].endswith('episode_9_seed3407.npz') for r in rows)
    groups = {}
    for r in rows:
        assert r['history_end_exclusive'] - r['history_start'] == r['H']
        assert r['query_index'] == r['history_end_exclusive']
        assert r['request']['reasoning_effort'] == 'medium'
        assert 'temperature' not in r['request']
        key = (r['model_alias'], r['H'], r['history_start'], r['shots'], r['repeat_id'])
        groups.setdefault(key, {})[r['reward_present']] = r
    for pair in groups.values():
        on, off = pair[True], pair[False]
        assert on['system_prompt'] == off['system_prompt']
        assert on['target_user_prompt'].count('  reward:') == on['H']
        assert '  reward:' not in off['target_user_prompt']
        # Every non-reward line and every original blank line is preserved.
        expected = ''.join(line for line in on['target_user_prompt'].splitlines(keepends=True)
                           if not line.startswith('  reward:'))
        assert off['target_user_prompt'] == expected
        assert off['user_prompt'][:off['target_offset']] == on['user_prompt'][:on['target_offset']]
        assert f'Step {on["query_index"]}:' not in on['target_user_prompt']
    for condition in {r['condition_id'] for r in rows}:
        repeats = [r for r in rows if r['condition_id']==condition]
        assert {r['repeat_id'] for r in repeats} == {1, 2, 3}
        assert len({r['user_prompt'] for r in repeats}) == 1


def test_fewshot_matches_08_verbatim(plan):
    snapshot = common.read_json(planning.HERE / 'data_prep/fewshot_p1_k4.json')
    source = common.read_json(planning.ROOT / snapshot['source_manifest'])
    q = next(q for q in source['queries'] if q['query_id']==snapshot['source_query_id'])
    assert snapshot['prefix'] == q['user_prompt'][:q['target_offset']]
    for row in plan['queries']:
        if row['shots']:
            assert row['example_ids'] == q['example_ids']
            assert row['user_prompt'] == snapshot['prefix'] + row['target_user_prompt']
        else:
            assert row['user_prompt'] == row['target_user_prompt']
            assert row['example_ids'] == []


def fake_client(monkeypatch, fail_at=None):
    calls = []
    class Client:
        def __init__(self, **kwargs):
            assert kwargs == dict(api_key='offline-test', max_retries=0, timeout=180.0)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == fail_at:
                raise RuntimeError('simulated interrupted request')
            raw = {'choices':[{'message':{'content':'predictions = [0.0]\n>>Final action bins: [5]'}}],
                   'usage':{'prompt_tokens':10,'completion_tokens':10,'total_tokens':20}}
            return SimpleNamespace(model_dump=lambda **kw:raw)
        def close(self):
            pass
    import openai
    monkeypatch.setattr(openai, 'OpenAI', Client)
    monkeypatch.setenv('OPENAI_API_KEY', 'ignored-environment-key')
    monkeypatch.setattr(execution.getpass, 'getpass', lambda prompt: 'offline-test')
    return calls


def small_plan(plan, count=6):
    result = deepcopy(plan)
    result['queries'] = result['queries'][:count]
    return result


def test_execute_score_resume_and_summary(plan, tmp_path, monkeypatch):
    manifest = small_plan(plan)
    planning.save_preview(manifest, tmp_path)
    calls = fake_client(monkeypatch)
    assert execution.execute(manifest, tmp_path) == 0
    assert len(calls) == 6
    records = common.read_jsonl(tmp_path / 'api/records.jsonl')
    assert len(records) == 6
    assert {r['source_experiment'] for r in records} == {'09_ablation_reward'}
    assert {r['repeat_id'] for r in records} == {1, 2, 3}
    assert all(r['action_value_prediction'] == [0.] for r in records)
    with pytest.raises(FileExistsError):
        execution.execute(manifest, tmp_path)
    monkeypatch.setattr(execution.getpass, 'getpass', lambda prompt: pytest.fail('Completed run must not prompt'))
    assert execution.execute(manifest, tmp_path, resume=True) == 0
    assert len(calls) == 6
    metrics, diff = analysis.summarize(tmp_path)
    assert len(metrics) == 2 and len(diff) == 1
    assert diff[0]['action_mae_off_minus_on'] == 0
    assert diff[0]['bin_accuracy_off_minus_on'] == 0


def test_error_is_never_resent_and_missing_difference_is_null(plan, tmp_path, monkeypatch):
    manifest = small_plan(plan)
    planning.save_preview(manifest, tmp_path)
    calls = fake_client(monkeypatch, fail_at=2)
    with pytest.raises(RuntimeError, match='simulated'):
        execution.execute(manifest, tmp_path)
    assert len(calls) == 2
    assert execution.execute(manifest, tmp_path, resume=True) == 2
    assert len(calls) == 6  # Only the four previously unattempted requests are sent.
    summary = common.read_json(tmp_path / 'api/summary.json')
    assert summary['unresolved_queries'] == 1
    assert summary['saved_responses'] == 5
    _, diff = analysis.summarize(tmp_path)
    assert diff[0]['action_mae_off_minus_on'] is None
    assert diff[0]['bin_accuracy_off_minus_on'] is None


def test_saved_response_can_be_rescored_without_api(plan, tmp_path, monkeypatch):
    manifest = small_plan(plan, 2)
    planning.save_preview(manifest, tmp_path)
    calls = fake_client(monkeypatch)
    execution.execute(manifest, tmp_path)
    (tmp_path / 'api/records.jsonl').write_text('')
    assert execution.execute(manifest, tmp_path, resume=True) == 0
    assert len(calls) == 2
    assert len(common.read_jsonl(tmp_path / 'api/records.jsonl')) == 2


def test_changed_plan_refuses_resume(plan, tmp_path, monkeypatch):
    manifest = small_plan(plan, 2)
    planning.save_preview(manifest, tmp_path)
    calls = fake_client(monkeypatch)
    execution.execute(manifest, tmp_path)
    altered = deepcopy(manifest)
    altered['queries'][0]['repeat_id'] = 99
    planning.save_preview(altered, tmp_path)
    with pytest.raises(ValueError, match='Resume plan differs'):
        execution.execute(altered, tmp_path, resume=True)
    assert len(calls) == 2


def test_dry_run_never_constructs_client(tmp_path, monkeypatch):
    calls = fake_client(monkeypatch)
    monkeypatch.setattr(planning, 'HERE', tmp_path)
    monkeypatch.setattr(planning, 'make_plan', lambda spec:{'queries':[]})
    monkeypatch.setattr(execution.getpass, 'getpass', lambda prompt: pytest.fail('Dry run must not prompt'))
    assert planning.run(entry.EXPERIMENT, ['--dry-run']) == 0
    assert calls == []
    assert not (tmp_path / 'results/api').exists()


def test_no_flags_executes_and_summarizes(plan, tmp_path, monkeypatch):
    manifest = small_plan(plan)
    calls = fake_client(monkeypatch)
    monkeypatch.setattr(planning, 'HERE', tmp_path)
    monkeypatch.setattr(planning, 'make_plan', lambda spec:manifest)
    assert planning.run(entry.EXPERIMENT, []) == 0
    assert len(calls) == 6
    assert (tmp_path / 'results/analysis/reward_differences.csv').exists()


def test_no_key_stops_before_any_request(plan, tmp_path, monkeypatch):
    manifest = small_plan(plan)
    planning.save_preview(manifest, tmp_path)
    calls = fake_client(monkeypatch)
    monkeypatch.setattr(execution.getpass, 'getpass', lambda prompt: '  ')
    with pytest.raises(ValueError, match='APIキーが空'):
        execution.execute(manifest, tmp_path)
    assert calls == []
    assert not (tmp_path / 'api').exists()
