import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent / '13_LAG')]
from commentary_experiment import runner, evaluation, execution, prompts


@pytest.fixture(scope='module')
def short_plan():
    return runner.make_plan(variant='tactical_short_v1')


def test_saved_baseline_and_exact_windows(short_plan):
    source = HERE / 'results/tactical_batch_v1/manifest.json'
    baseline = json.loads(source.read_text())
    spec = short_plan['prompt_spec']
    comparison = spec['comparison']
    assert comparison['source_manifest_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert comparison['source_plan_id'] == baseline['plan_id']
    assert short_plan['counts'] == baseline['counts']
    assert short_plan['request_options'] == baseline['request_options']
    assert [q['model'] for q in short_plan['queries']] == [q['model'] for q in baseline['queries']]
    for sid, sample in short_plan['samples'].items():
        old = baseline['samples'][sid]
        assert sample['window'] == old['window']
        assert old['messages'][0]['content'] == comparison['baseline_templates']['system']
        window = sample['window']
        data = '\n'.join(json.dumps(b, ensure_ascii=False, allow_nan=False, separators=(',', ':')) for b in window['blocks'])
        fields = {k: v for k, v in window.items() if k != 'blocks'}
        assert comparison['baseline_templates']['user'].format(**fields, window_data=data) == old['messages'][1]['content']
        new_user = sample['messages'][1]['content']
        assert f"Center step t: {window['center_step']}" in new_user
        assert 'annotat' not in json.dumps(sample['messages']).lower()
        assert 'change point' not in json.dumps(sample['messages']).lower()
        assert 'around T' not in json.dumps(sample['messages'])
        assert new_user.split('BEGIN WINDOW DATA')[1].split('END WINDOW DATA')[0] == old['messages'][1]['content'].split('BEGIN WINDOW DATA')[1].split('END WINDOW DATA')[0]
    # Input definitions and missile constraints must survive the focused edits.
    before, after = comparison['baseline_templates']['system'], spec['templates']['system']['content']
    assert before.split('# ENVIRONMENT')[1].split('# TIME AND DATA AVAILABILITY')[0] == after.split('# ENVIRONMENT')[1].split('# TIME AND DATA AVAILABILITY')[0]
    assert before.split('# MISSILE INFORMATION')[1].split('# PRIMARY TASK')[0] == after.split('# MISSILE INFORMATION')[1].split('# PRIMARY TASK')[0]
    assert '90–160' not in after
    assert short_plan['prompt_version'] == spec['version']
    assert all('/' + spec['variant'] in q['condition_id'] for q in short_plan['queries'])
    for prefix in ('Your task is to ', 'The center step t '):
        assert next(line for line in before.splitlines() if line.startswith(prefix)) == next(line for line in after.splitlines() if line.startswith(prefix))


def test_frozen_snapshots_and_overwrite_protection(short_plan, tmp_path, monkeypatch):
    execution.save_plan(short_plan, tmp_path)
    frozen = (tmp_path / 'prompts/templates/system.txt').read_text()
    # Saving an already frozen plan must not consult the current source templates.
    monkeypatch.setattr(prompts, 'get_prompt_spec', lambda *_: pytest.fail('Read live template'))
    execution.save_plan(short_plan, tmp_path)
    assert frozen == short_plan['prompt_spec']['templates']['system']['content']
    assert (tmp_path / 'prompts/changes/user.diff').read_text() == short_plan['prompt_spec']['comparison']['diffs']['user']
    altered = dict(short_plan, prompt_version='another-version')
    altered['plan_id'] = execution.digest({k: v for k, v in altered.items() if k != 'plan_id'})
    with pytest.raises(FileExistsError):
        execution.save_plan(altered, tmp_path)
    (tmp_path / 'prompts/templates/system.txt').write_text('edited')
    with pytest.raises(ValueError, match='snapshot was edited'):
        execution.save_plan(short_plan, tmp_path)


def test_dry_run_uses_registered_directory(short_plan, tmp_path, monkeypatch, capsys):
    variant = short_plan['prompt_variant']
    monkeypatch.setattr(runner, 'HERE', tmp_path)
    monkeypatch.setattr(runner, 'make_plan', lambda models, repeats, selected: short_plan if selected == variant else pytest.fail(selected))
    monkeypatch.setattr(runner.batch, 'prepare', lambda plan, directory: {'prepared': plan['counts']['requests']})
    monkeypatch.setattr(runner.batch, 'local_status', lambda state: state)
    monkeypatch.setattr(runner.batch, 'execute', lambda *a, **k: pytest.fail('Unexpected API execution'))
    assert runner.run(['--prompt-variant', variant]) == 0
    directory = tmp_path / 'results' / short_plan['prompt_spec']['default_output']
    assert execution.read_json(directory / 'manifest.json') == short_plan
    assert len(list((directory / 'windows').glob('*.json'))) == 5
    assert 'No API calls made.' in capsys.readouterr().out


def test_export_counts_without_truncating(short_plan, tmp_path):
    execution.save_plan(short_plan, tmp_path)
    (tmp_path / 'api').mkdir()
    narratives = {'A0100': ' '.join(['word'] * 91), 'A0200': ' '.join(['word'] * 90),
                  'B0100': 'Continued flight.', 'B0200': 'Inactive.'}
    raw = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(narratives)}}]}
    execution.append_journal(tmp_path / 'api/responses.jsonl', {'query_id': short_plan['queries'][0]['query_id'], 'raw_response': raw})
    evaluation.evaluate_saved(short_plan, tmp_path)
    output = execution.read_json(tmp_path / 'evaluation/responses.json')
    assert output[0]['intention_format']['narratives'] == narratives
    assert output[0]['length_metrics']['words_by_aircraft']['A0100'] == 91
    assert output[0]['length_metrics']['over_limit_by_aircraft'] == {'A0100': True, 'A0200': False, 'B0100': False, 'B0200': False}
    assert output[1]['length_metrics'] is None


def test_viewer_uses_manifest_not_directory_name(short_plan, tmp_path):
    spec = importlib.util.spec_from_file_location('short_viewer', HERE / 'build_results_viewer.py')
    viewer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(viewer)
    directory = tmp_path / 'arbitrary-name'
    execution.save_plan(short_plan, directory)
    data = viewer.load_data(directory)
    assert data['prompt_variant'] == short_plan['prompt_variant']
    assert data['prompt_description'] == short_plan['prompt_spec']['description']
    assert data['prompt_template_sha256']['user'] == short_plan['prompt_spec']['templates']['user']['sha256']
    assert data['counts'] == {'unstarted': 15}
    for name in ('commentary_batch_v1', 'tactical_batch_v1'):
        old = viewer.load_data(HERE / 'results' / name)
        assert old['prompt_template_sha256'] == {}
        assert old['prompt_description']


def test_stale_diff_is_rejected(tmp_path, monkeypatch):
    import shutil
    for role in ('system', 'user'):
        shutil.copy(prompts.HERE / f'tactical_short_v1_{role}.txt', tmp_path)
    shutil.copytree(prompts.HERE / 'prompt_changes', tmp_path / 'prompt_changes')
    monkeypatch.setattr(prompts, 'HERE', tmp_path)
    (tmp_path / 'tactical_short_v1_user.txt').write_text('Changed without updating review artifacts')
    with pytest.raises(ValueError, match='Stale user diff'):
        prompts.get_prompt_spec('tactical_short_v1')
