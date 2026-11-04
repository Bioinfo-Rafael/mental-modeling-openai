import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent/'13_LAG')]
from commentary_experiment import runner, execution as ex, evaluation, timing


@pytest.fixture(scope='module')
def plan():
    return runner.make_plan(models=('luna',), variant='tactical_short_counttime_v1')


def test_two_outputs_same_input(plan):
    old = ex.read_json(HERE/'results/tactical_short_batch_v2/manifest.json')
    assert plan['counts'] == dict(samples=5, requests=5, models=1, repeats=1)
    assert [q['model'] for q in plan['queries']] == ['gpt-5.6-luna']*5
    for sid, sample in plan['samples'].items():
        assert sample['window'] == old['samples'][sid]['window']
    raw = {'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'A0100':'Purpose A.','A0200':'Purpose B.'})}}]}
    assert evaluation.parse_intentions(raw, ('A0100','A0200'))['valid']
    assert not evaluation.parse_intentions(raw)['valid']
    raw['choices'][0]['message']['content'] = json.dumps(dict(A0100='A',A0200='B',B0100='C',B0200='D'))
    assert not evaluation.parse_intentions(raw, ('A0100','A0200'))['valid']


def test_exact_timing_excludes_journaling_and_serializing_and_resume(plan, tmp_path, monkeypatch):
    ex.save_plan(plan, tmp_path)
    now = [0.0]
    append = ex.append_journal
    def slow_append(*args):
        now[0] += 100
        return append(*args)
    monkeypatch.setattr(ex, 'append_journal', slow_append)
    monkeypatch.setattr(ex.time, 'perf_counter', lambda: now[0])
    calls = []
    class Result:
        _request_id = 'test-request'
        def model_dump(self, mode):
            now[0] += 50
            return {'choices':[{'finish_reason':'stop','message':{'content':'{"A0100":"Goal A.","A0200":"Goal B."}'}}], 'usage':{}}
    def create(**kwargs):
        calls.append(kwargs)
        assert len(ex.read_journal(tmp_path/'api/requests.jsonl')) == len(calls)
        assert len(ex.read_journal(tmp_path/'api/responses.jsonl')) == len(calls)-1
        now[0] += 2
        return Result()
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=lambda:None)
    summary = ex.execute(plan, tmp_path, repository_root=runner.ROOT, client_factory=lambda:client)
    assert summary['status']=='complete'
    assert len(calls)==5
    assert [r['elapsed_seconds'] for r in ex.read_journal(tmp_path/'api/responses.jsonl')]==[2.0]*5
    timing_summary=timing.export(plan,tmp_path)
    assert timing_summary['mean_seconds']==2 and timing_summary['valid_format_responses']==5
    ex.execute(plan,tmp_path,repository_root=runner.ROOT,resume=True,
               client_factory=lambda:pytest.fail('Must not send again'))


def test_failure_is_measured_but_not_resent(plan, tmp_path):
    ex.save_plan(plan,tmp_path)
    def create(**kwargs):
        raise RuntimeError('Connection failed')
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),close=lambda:None)
    with pytest.raises(RuntimeError,match='Connection failed'):
        ex.execute(plan,tmp_path,repository_root=runner.ROOT,client_factory=lambda:client)
    failed=ex.read_journal(tmp_path/'api/responses.jsonl')[0]
    assert failed['elapsed_seconds']>=0 and failed['raw_response'] is None
    summary=timing.export(plan,tmp_path)
    assert summary['mean_seconds'] is None and summary['unattempted']==4


def test_cli_dry_run_does_not_prepare_batches(plan, tmp_path, monkeypatch):
    monkeypatch.setattr(runner,'HERE',tmp_path)
    monkeypatch.setattr(runner,'make_plan',lambda *args:plan)
    monkeypatch.setattr(runner.batch,'prepare',lambda *args:pytest.fail('Unexpected batch'))
    monkeypatch.setattr(runner.ex,'execute',lambda *args,**kwargs:pytest.fail('Unexpected API call'))
    assert runner.run(['--prompt-variant','tactical_short_counttime_v1','--dry-run'])==0
    path=tmp_path/'results/tactical_short_CountTime_v1'
    assert not (path/'api').exists()
    assert ex.read_json(path/'evaluation/timing.json')['summary']['unattempted']==5
