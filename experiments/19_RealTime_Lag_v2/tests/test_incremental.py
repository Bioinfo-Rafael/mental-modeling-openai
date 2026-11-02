import importlib
from pathlib import Path

import pytest

w = importlib.import_module('experiments.19_RealTime_Lag_v2.realtime_experiment.workflow')
BASE = w.ROOT / 'experiments/19_RealTime_LAG'


def test_targets_partition_episode_including_initial_and_final_window(tmp_path, monkeypatch):
    monkeypatch.setattr(w, 'RESULTS', tmp_path / 'results')
    directory = w.create_plan('boundaries')
    m = w.verify(directory)
    assert len(m['samples']) == 62 and len(m['queries']) == 434
    first, second, last = m['samples'][0], m['samples'][1], m['samples'][-1]
    assert (first['previous_end_step'], first['target_start_step'], first['target_end_step']) == (None, 0, 39)
    assert (second['previous_end_step'], second['target_start_step'], second['target_end_step']) == (39, 40, 44)
    assert (last['previous_end_step'], last['target_start_step'], last['target_end_step']) == (339, 340, 340)
    covered = []
    for sample in m['samples']:
        window = w.read_json(directory / 'inputs' / f"{sample['id']}.json")
        assert len(window['blocks']) == 40
        assert max(b['step'] for b in window['blocks']) == sample['end_step']
        covered.extend(range(sample['target_start_step'], sample['target_end_step'] + 1))
    assert covered == list(range(341))  # no repeated target steps and no omitted steps
    assert all(s['target_time_points'] == 5 for s in m['samples'][1:-1])
    q = next(q for q in m['queries'] if q['stage'] == 'gate' and q['source_end_step'] == 54)
    body = w.request_for(directory, q)
    assert 'Previous window end step: 49' in body['input'][1]['content']
    assert 'TARGET INTERVAL: steps 50 through 54, inclusive' in body['input'][1]['content']
    assert 'Window: steps 15 through 54' in body['input'][1]['content']
    assert 'becomes clearer inside it' in body['input'][0]['content']
    assert 'A change may occur anywhere in the window' not in body['input'][0]['content']
    # The schedule boundary is fixed in the plan, independent of later NONE/late/error results.
    assert q['previous_end_step'] == 49


def test_goal_prompts_choices_and_runtime_settings_unchanged():
    for folder in ('choices', 'prompts/goal_v001'):
        for source in (BASE / folder).rglob('*'):
            if source.is_file():
                assert source.read_bytes() == (w.HERE / source.relative_to(BASE)).read_bytes()
    old = w.read_json(BASE / 'configs/experiment.json')
    new = w.read_json(w.HERE / 'configs/experiment.json')
    for key in ('goal', 'deadline', 'replay', 'dt_s', 'history_points', 'stride_steps'):
        assert old[key] == new[key]


def test_cache_namespace_preserves_original_run(tmp_path, monkeypatch):
    monkeypatch.setattr(w, 'RESULTS', tmp_path / 'results')
    monkeypatch.setattr(w, 'VIEWER', tmp_path / 'viewer')
    w.write_json(w.VIEWER / 'index.json', [{'id': 'same_name', 'created_at': '2026-01-01', 'url': '/original.json'}])
    directory = w.create_plan('same_name', 1, 1)
    w.export(directory)
    index = w.read_json(w.VIEWER / 'index.json')
    assert {r['id'] for r in index} == {'same_name', '19_RealTime_Lag_v2__same_name'}
    assert next(r for r in index if r['id'] == 'same_name')['url'] == '/original.json'


def test_target_boundary_rejects_non_incremental_schedule():
    for previous in (9, 54, 55):
        with pytest.raises(ValueError):
            w.target_fields({'start_step': 15, 'end_step': 54}, previous)
