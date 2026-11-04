"""Verify comparisons against executed inputs and frozen review artifacts."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent / '13_LAG')]
from commentary_experiment import runner, prompts, execution


@pytest.fixture(scope='module')
def plan():
    return runner.make_plan(variant='tactical_short_v2')


def test_both_comparisons_match_saved_messages_and_same_data(plan):
    spec = plan['prompt_spec']
    assert set(spec['comparisons']) == {'from_tactical_batch_v1', 'from_tactical_short_batch_v1'}
    assert spec['default_output'] == 'tactical_short_batch_v2'
    assert spec['length']['max_words'] == 55
    for comparison in spec['comparisons'].values():
        path = HERE / 'results' / comparison['source_run'] / 'manifest.json'
        old = json.loads(path.read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == comparison['source_manifest_sha256']
        assert old['plan_id'] == comparison['source_plan_id']
        assert plan['counts'] == old['counts']
        assert plan['request_options'] == old['request_options']
        assert [q['model'] for q in plan['queries']] == [q['model'] for q in old['queries']]
        for sid, sample in plan['samples'].items():
            previous = old['samples'][sid]
            assert sample['window'] == previous['window']
            window = sample['window']
            data = '\n'.join(json.dumps(b, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
                             for b in window['blocks'])
            fields = {k: v for k, v in window.items() if k != 'blocks'}
            assert comparison['baseline_templates']['system'] == previous['messages'][0]['content']
            assert comparison['baseline_templates']['user'].format(**fields, window_data=data) == previous['messages'][1]['content']
            messages = json.dumps(sample['messages']).lower()
            assert 'annotat' not in messages
            assert 'change point' not in messages
    short = spec['comparisons']['from_tactical_short_batch_v1']
    assert short['diffs']['user'] == ''
    assert spec['templates']['user']['content'] == short['baseline_templates']['user']
    before, after = short['baseline_templates']['system'], spec['templates']['system']['content']
    assert before.split('# ENVIRONMENT')[1].split('# PRIMARY TASK')[0] == after.split('# ENVIRONMENT')[1].split('# PRIMARY TASK')[0]
    for prefix in ('Your task is to ', 'The center step t '):
        assert next(l for l in before.splitlines() if l.startswith(prefix)) == next(l for l in after.splitlines() if l.startswith(prefix))


def test_both_comparisons_are_frozen_and_guarded(plan, tmp_path, monkeypatch):
    execution.save_plan(plan, tmp_path)
    monkeypatch.setattr(prompts, 'get_prompt_spec', lambda *_: pytest.fail('Read mutable source'))
    execution.save_plan(plan, tmp_path)
    for name, comparison in plan['prompt_spec']['comparisons'].items():
        for role in ('system', 'user'):
            assert (tmp_path / f'prompts/changes/{name}/{role}.diff').read_text() == comparison['diffs'][role]
    snapshot = tmp_path / 'prompts/changes/from_tactical_short_batch_v1/system.diff'
    snapshot.write_text('Changed review artifact')
    with pytest.raises(ValueError, match='snapshot was edited'):
        execution.save_plan(plan, tmp_path)


def test_stale_second_comparison_is_rejected(tmp_path, monkeypatch):
    import shutil
    for role in ('system', 'user'):
        shutil.copy(prompts.HERE / f'tactical_short_v2_{role}.txt', tmp_path)
    shutil.copytree(prompts.HERE / 'prompt_changes/tactical_short_v2', tmp_path / 'prompt_changes/tactical_short_v2')
    monkeypatch.setattr(prompts, 'HERE', tmp_path)
    diff = tmp_path / 'prompt_changes/tactical_short_v2/from_tactical_batch_v1/system.diff'
    diff.write_text('Stale diff')
    with pytest.raises(ValueError, match='Stale system diff'):
        prompts.get_prompt_spec('tactical_short_v2')
