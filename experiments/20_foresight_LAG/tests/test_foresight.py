from copy import deepcopy
from importlib import import_module
import json
import shutil

import pytest

common = import_module('experiments.20_foresight_LAG.foresight_experiment.common')
planning = import_module('experiments.20_foresight_LAG.foresight_experiment.planning')
dataset = import_module('experiments.20_foresight_LAG.foresight_experiment.dataset')
evaluation = import_module('experiments.20_foresight_LAG.foresight_experiment.evaluation')
viewer = import_module('experiments.20_foresight_LAG.foresight_experiment.viewer')
cli = import_module('experiments.20_foresight_LAG.foresight_experiment.cli')


@pytest.fixture(scope='module')
def plan():
    return planning.make_plan()


def test_complete_common_events_and_no_repeat(plan):
    assert plan['counts']['events'] == 6
    assert plan['counts']['samples'] == 18
    assert plan['counts']['requests'] == 54
    assert {q['repeat_id'] for q in plan['queries']} == {1}
    assert len({q['query_id'] for q in plan['queries']}) == 54
    assert len({q['condition_id'] for q in plan['queries']}) == 9
    assert len(plan['excluded_events']) == 1
    assert plan['excluded_events'][0]['event']['step'] == 311
    assert plan['excluded_events'][0]['missing_steps'] == [341]


def test_nested_windows_fixed_target_and_no_label_leak(plan):
    systems = set()
    for sample in plan['samples'].values():
        t, n = sample['target_step'], sample['offset']
        w = sample['window']
        assert sample['target_interval'] == [t, t + 30]
        assert [b['step'] for b in w['blocks']] == list(range(t - 30, t + n + 1))
        assert w['time_points'] == 31 + n
        assert w['elapsed_s'] == round((30 + n) * .2, 1)
        assert w['blocks'][-1]['relative_step'] == 0
        systems.add(sample['messages'][0]['content'])
        user = sample['messages'][1]['content']
        raw = user.split('BEGIN OBSERVATION DATA\n\n')[1].split('\n\nEND OBSERVATION DATA')[0]
        assert [json.loads(line) for line in raw.splitlines()] == w['blocks']
        assert user.count('Target interval:') == 1
        assert f'Start step T: {t}' in user
        for e in plan['annotations']['events']:
            assert e['label'] not in user
        for forbidden in ('annotation', 'change point', 'condition_kind', 'target_aircraft', 'offset:'):
            assert forbidden not in user.lower()
        assert sample['condition_kind'] == ('hindsight_reference' if n == 30 else 'forecast')
        reference = plan['samples'][f"{sample['event']['id']}_n+30"]['window']['blocks']
        # c-relative coordinates change; physical records must not.
        strip = lambda b: {k: v for k, v in b.items() if k != 'relative_step'}
        assert [strip(b) for b in w['blocks']] == [strip(b) for b in reference[:len(w['blocks'])]]
    assert len(systems) == 1


def test_records_match_experiment_17(plan, monkeypatch):
    monkeypatch.syspath_prepend(str(common.ROOT / 'experiments/13_LAG'))
    legacy = import_module('experiments.17_commentary_LAG.commentary_experiment.dataset')
    frame = common.lag.load_episode(common.lag.PROCESSED / f"{plan['config']['episode_id']}.csv")
    for t in [46, 143, 243]:
        original = legacy.build_window(frame, t)['blocks']
        new = plan['samples'][f'e{t}_n+30']['window']['blocks']
        strip = lambda b: {k: v for k, v in b.items() if k != 'relative_step'}
        assert [strip(b) for b in new] == [strip(b) for b in original]


def test_internal_gap_excludes_all_conditions_not_truncated(plan):
    frame = common.lag.load_episode(common.lag.PROCESSED / f"{plan['config']['episode_id']}.csv")
    samples, excluded = dataset.build_samples(frame[frame.step.ne(50)], plan['annotations'], plan['config'])
    assert not any(s['event']['id'] in ('e46', 'e49') for s in samples.values())
    assert {e['event']['id'] for e in excluded} >= {'e46', 'e49', 'e311'}


def test_baseline_technical_sections_unchanged(plan):
    spec = plan['prompt_spec']
    old = spec['comparison']['baseline_templates']['system']
    new = spec['templates']['system']['content']
    assert old.split('# ENVIRONMENT')[1].split('# TIME AND DATA AVAILABILITY')[0] == new.split('# ENVIRONMENT')[1].split('# TIME AND DATA AVAILABILITY')[0]
    assert old.split('Data conventions:')[1].split('# PRIMARY TASK:')[0] == new.split('Data conventions:')[1].split('# TASK')[0]
    for phrase in ['Use ALL 61', 'Do not infer events outside', 'supporting circumstance integrated', 'second short sentence']:
        assert phrase not in new
    assert 'supporting evidence or reasoning' in new
    assert 'not\nan unseen-future prediction' in new


def test_stale_diff_and_changed_prompt_refused(tmp_path, monkeypatch):
    shutil.copytree(common.HERE / 'prompts', tmp_path / 'prompts')
    monkeypatch.setattr(planning, 'HERE', tmp_path)
    monkeypatch.setattr(planning, 'ROOT', tmp_path)
    folder = tmp_path / 'prompts/foresight_short_v1'
    diff = folder / 'changes/system.diff'
    original = diff.read_text()
    diff.write_text('stale')
    with pytest.raises(ValueError, match='Stale system diff'):
        planning.prompt_spec()
    diff.write_text(original)
    (folder / 'system.txt').write_text('modified')
    with pytest.raises(ValueError, match='Proposal system hash'):
        planning.prompt_spec()


def test_save_idempotent_and_changed_plan_refused(plan, tmp_path):
    planning.save_plan(plan, tmp_path)
    planning.save_plan(plan, tmp_path)
    assert not (tmp_path / 'api/requests.jsonl').exists()
    assert (tmp_path / 'prompts/changes/system.diff').read_text() == plan['prompt_spec']['comparison']['diffs']['system']
    changed = deepcopy(plan)
    changed['config']['repeats'] = 2
    changed['plan_id'] = common.ex.digest({k: v for k, v in changed.items() if k != 'plan_id'})
    with pytest.raises(FileExistsError):
        planning.save_plan(changed, tmp_path)
    with pytest.raises(ValueError):
        common.run_directory('../../escape')


def test_offline_exports_and_safe_html(plan, tmp_path):
    planning.save_plan(plan, tmp_path)
    rows, summary = evaluation.export(plan, tmp_path)
    assert summary['statuses'] == {'unsubmitted': 54}
    assert all(row['narratives'] is None for row in rows)
    malicious = '</script><script>alert(1)</script>'
    rows[0]['error'] = malicious
    page = viewer.build_viewer(plan, rows, summary, tmp_path).read_text()
    assert malicious not in page
    embedded = page.split('<script id="dataset" type="application/json">')[1].split('</script>')[0]
    assert json.loads(embedded)['rows'][0]['error'] == malicious
    assert '__DATA__' not in page
    template = tmp_path / 'report/review_template.csv'
    template.write_text('human edits', encoding='utf-8')
    evaluation.export(plan, tmp_path)
    assert template.read_text() == 'human edits'


def test_mock_batch_submit_collect_and_no_resend(plan, tmp_path):
    FakeClient = import_module('experiments.17_commentary_LAG.tests.test_batch').FakeClient
    client = FakeClient()
    planning.save_plan(plan, tmp_path)
    common.batch.execute(plan, tmp_path, repository_root=common.ROOT, client_factory=lambda: client)
    assert len(client.created) == 3
    client.finish()
    for file_id, text in list(client.outputs.items()):
        if not text:
            continue
        outputs = [json.loads(line) for line in text.splitlines()]
        for row in outputs:
            row['response']['body']['choices'][0].update(finish_reason='stop', message={'content': json.dumps({a: 'A concise forecast.' for a in ['A0100', 'A0200', 'B0100', 'B0200']})})
        client.outputs[file_id] = '\n'.join(map(json.dumps, outputs))
    common.batch.execute(plan, tmp_path, repository_root=common.ROOT, submit_new=False, client_factory=lambda: client)
    rows, summary = evaluation.export(plan, tmp_path)
    assert summary['statuses'] == {'valid': 54}
    assert all(r['length_metrics']['words_by_aircraft']['A0100'] == 3 for r in rows)
    saved = (tmp_path / 'api/responses.jsonl').read_bytes()
    common.batch.execute(plan, tmp_path, repository_root=common.ROOT, client_factory=lambda: client)
    assert len(client.created) == 3
    assert saved == (tmp_path / 'api/responses.jsonl').read_bytes()
    viewer.build_viewer(plan, rows, summary, tmp_path)


def test_errors_are_not_valid_predictions(plan, tmp_path):
    planning.save_plan(plan, tmp_path)
    state = common.batch.load_state(plan, tmp_path)
    q = plan['queries'][0]
    job = next(j for j in state['jobs'] if q['query_id'] in j['query_ids'])
    common.ex.append_journal(tmp_path / 'api/requests.jsonl', dict(query_id=q['query_id'], request=common.ex.request_for(plan, q), batch_job_key=job['job_key']))
    common.ex.append_journal(tmp_path / 'api/responses.jsonl', dict(query_id=q['query_id'], raw_response={'choices': [{'finish_reason': 'length', 'message': {'content': '{'}}]}, error=None))
    rows, summary = evaluation.export(plan, tmp_path)
    assert rows[0]['status'] == 'format_error'
    assert rows[0]['raw_response'] is not None
    assert rows[0]['narratives'] is None
    assert summary['statuses'] == {'format_error': 1, 'unsubmitted': 53}


def test_source_changes_block_submission(plan, tmp_path):
    changed = deepcopy(plan)
    changed['source_sha256']['experiments/20_foresight_LAG/run.py'] = 'wrong'
    changed['plan_id'] = common.ex.digest({k: v for k, v in changed.items() if k != 'plan_id'})
    common.ex.write_json(tmp_path / 'manifest.json', changed)
    with pytest.raises(ValueError, match='Source changed'):
        common.batch.validate_sources(changed, tmp_path, common.ROOT)


def test_cli_wait_and_missing_run_rejected():
    with pytest.raises(SystemExit):
        cli.run(['plan', '--wait'])
    with pytest.raises(SystemExit):
        cli.run(['collect', '--run', 'nonexistent-test-run'])


def test_targets_preserves_inputs_and_changes_requests(plan, tmp_path):
    targets = planning.make_plan('targets_v1')
    assert targets['counts'] == plan['counts']
    assert targets['prompt_variant'] == 'targets_v1'
    assert targets['prompt_spec']['length']['max_words'] == 80
    for sid, sample in targets['samples'].items():
        assert sample['window'] == plan['samples'][sid]['window']
        assert sample['messages'][1] == plan['samples'][sid]['messages'][1]
        assert sample['messages'][0] != plan['samples'][sid]['messages'][0]
    assert {q['request_sha256'] for q in targets['queries']}.isdisjoint(
        q['request_sha256'] for q in plan['queries'])
    planning.save_plan(targets, tmp_path)
    rows, summary = evaluation.export(targets, tmp_path)
    page = viewer.build_viewer(targets, rows, summary, tmp_path).read_text()
    assert summary['statuses'] == {'unsubmitted': 54}
    assert '上限55語' not in page
    with pytest.raises(ValueError, match='Unknown prompt variant'):
        planning.prompt_spec('../invalid')
