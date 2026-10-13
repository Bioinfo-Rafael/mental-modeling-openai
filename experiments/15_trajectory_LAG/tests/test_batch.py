"""Batch-from-start contracts; all API objects are local fakes."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent / "13_LAG"), str(HERE.parents[1] / "upstream/LLM-Xavier")]
from trajectory_experiment import batch, execution as ex, evaluation, prompts
from lag_data.definitions import STATE_FEATURES, ACTIONS


def obj(data):
    return SimpleNamespace(**data, model_dump=lambda **kwargs: deepcopy(data))


@pytest.fixture
def run(tmp_path):
    sample = dict(messages=[dict(role='user', content='Predict ten state/action pairs')],
                  episode_id='episode', target_id='A0100', task='trajectory', H=20,
                  start=0, query_step=20, target_step=20, target_steps=list(range(20, 30)),
                  horizon=10, observation_scope='self', sample_number=0, problem_id='problem',
                  ground_truth=[dict(step=t, state=[0.] * 10, action=[1, 2, 0]) for t in range(20, 30)],
                  prediction_fields=dict(action=list(ACTIONS[:3]), state=list(STATE_FEATURES)))
    plan = dict(samples={'sample': sample}, queries=[], request_options={'reasoning_effort': 'medium'},
                source_sha256={}, counts=dict(samples=1, requests=10))
    for i in range(10):
        alias = ('sol', 'terra', 'luna')[i % 3]
        q = dict(sample_id='sample', model=f'gpt-5.6-{alias}', model_alias=alias,
                 repeat_id=i, condition_id=alias)
        q['request_sha256'] = ex.digest(ex.request_for(plan, q))
        q['query_id'] = ex.digest(q)
        plan['queries'].append(q)
    plan['plan_id'] = ex.digest(plan)
    ex.write_json(tmp_path / 'manifest.json', plan)
    return plan, tmp_path


class FakeClient:
    def __init__(self):
        self.inputs, self.outputs, self.remote, self.created = {}, {}, {}, []
        self.fail_create = False
        self.closed = False
        self.files = SimpleNamespace(create=self.upload, content=lambda id: SimpleNamespace(text=self.outputs[id]))
        self.batches = SimpleNamespace(create=self.create, retrieve=lambda id: obj(self.remote[id]),
                                       list=lambda **kw: [obj(x) for x in self.remote.values()])

    def close(self):
        self.closed = True

    def upload(self, file, purpose):
        assert purpose == 'batch'
        id = f'file_{len(self.inputs)}'
        self.inputs[id] = [json.loads(line) for line in file.read().decode().splitlines()]
        return SimpleNamespace(id=id)

    def create(self, **kwargs):
        id = f'batch_{len(self.remote)}'
        record = dict(id=id, status='in_progress', **kwargs)
        self.remote[id] = record
        self.created.append(id)
        if self.fail_create:
            raise TimeoutError('Connection lost AFTER remote acceptance')
        return obj(record)

    def finish(self, *, fail=False, omit=False):
        for id, remote in self.remote.items():
            output_id, error_id = id + '_out', id + '_err'
            outputs, errors = [], []
            for i, row in enumerate(reversed(self.inputs[remote['input_file_id']])):
                if omit and i == 0:
                    continue
                result = dict(custom_id=row['custom_id'], error=None, response=dict(status_code=200,
                              body={'choices': [{'message': {'content': prompts.TRAJECTORY_MARKER + ' ' + json.dumps([dict(step=t, state=[0.] * 10, action=[1, 2, 0]) for t in range(20, 30)])}}],
                                    'usage': {'prompt_tokens': 100, 'completion_tokens': 50}}))
                if fail and i == 0:
                    result.update(response=None, error={'code': 'batch_expired'})
                    errors.append(result)
                else:
                    outputs.append(result)
            self.outputs[output_id] = '\n'.join(map(json.dumps, outputs))
            self.outputs[error_id] = '\n'.join(map(json.dumps, errors))
            remote.update(status='expired' if fail or omit else 'completed',
                          output_file_id=output_id, error_file_id=error_id)


def test_prepare_partitions_whole_plan_without_sending(run):
    plan, directory = run
    state = batch.prepare(plan, directory, max_requests=2)
    assert sum(len(j['query_ids']) for j in state['jobs']) == 10
    assert len(state['jobs']) == 6
    assert all(len(j['query_ids']) <= 2 for j in state['jobs'])
    assert not (directory / 'api/requests.jsonl').exists()
    assert batch.load_state(plan, directory) == state
    batch.validate_sources(plan, directory, HERE.parents[1])


def test_prepare_rejects_existing_sync_run(run):
    plan, directory = run
    ex.write_json(directory / 'api/identity.json', {'plan_id': plan['plan_id']})
    with pytest.raises(ValueError, match='synchronous run'):
        batch.prepare(plan, directory)


def test_full_lifecycle_out_of_order_and_idempotent_collection(run):
    plan, directory = run
    state = batch.prepare(plan, directory, max_requests=2)
    client = FakeClient()
    batch.submit(client, plan, directory, state)
    assert len(client.created) == 3  # At most one active shard per model.
    batch.submit(client, plan, directory, state)
    assert len(client.created) == 3
    client.finish()
    batch.collect(client, plan, directory, state)
    state = batch.load_state(plan, directory)
    batch.submit(client, plan, directory, state)
    assert len(client.created) == 6
    client.finish()
    batch.collect(client, plan, directory, state)
    before = (directory / 'api/responses.jsonl').read_bytes()
    batch.collect(client, plan, directory, state)
    assert (directory / 'api/responses.jsonl').read_bytes() == before
    summary = batch.summarize(plan, directory, state)
    assert summary['responses_saved'] == 10 and summary['status'] == 'complete'
    assert len(ex.read_journal(directory / 'api/requests.jsonl')) == 10
    assert len(ex.read_journal(directory / 'api/responses.jsonl')) == 10
    records = ex.read_json(directory / 'analysis' / evaluation.SCORING_VERSION / 'per_query.json')
    assert sum(r['parse_success'] for r in records) == 10
    assert all(r['elapsed_seconds'] is None for r in records)


def test_lost_create_response_recovers_without_resubmit(run):
    plan, directory = run
    state = batch.prepare(plan, directory)
    client = FakeClient()
    client.fail_create = True
    with pytest.raises(TimeoutError):
        batch.submit(client, plan, directory, state)
    state = batch.load_state(plan, directory)
    assert state['jobs'][0]['status'] == 'submitting'
    batch.collect(client, plan, directory, state)
    assert state['jobs'][0]['batch_id'] == 'batch_0'
    assert len(client.created) == 1


def test_unknown_submission_never_automatically_retries(run):
    plan, directory = run
    state = batch.prepare(plan, directory)
    client = FakeClient()
    state['jobs'][0].update(status='submitting', file_id='file_unknown')
    with pytest.raises(RuntimeError, match='No automatic resend'):
        batch.collect(client, plan, directory, state)
    assert client.created == []


@pytest.mark.parametrize('omit', [False, True])
def test_expired_partial_results_saved_and_not_resent(run, omit):
    plan, directory = run
    state = batch.prepare(plan, directory)
    client = FakeClient()
    batch.submit(client, plan, directory, state)
    client.finish(fail=not omit, omit=omit)
    batch.collect(client, plan, directory, state)
    summary = batch.summarize(plan, directory, state)
    assert summary['unresolved'] == 3
    assert summary['responses_saved'] == 7
    assert len(ex.read_journal(directory / 'api/responses.jsonl')) == 10
    batch.collect(client, plan, directory, state)
    assert len(client.created) == 3


def test_duplicate_output_rejected_before_import(run):
    plan, directory = run
    state = batch.prepare(plan, directory)
    client = FakeClient()
    batch.submit(client, plan, directory, state)
    client.finish()
    key = 'batch_0_out'
    client.outputs[key] += '\n' + client.outputs[key].splitlines()[0]
    with pytest.raises(ValueError, match='duplicate custom_id'):
        batch.collect(client, plan, directory, state)
    assert len(ex.read_journal(directory / 'api/responses.jsonl')) == 0


def test_partial_reservation_and_partial_import_can_resume(run):
    plan, directory = run
    state = batch.prepare(plan, directory)
    client = FakeClient()
    job = state['jobs'][0]
    qid = job['query_ids'][0]
    q = next(q for q in plan['queries'] if q['query_id'] == qid)
    ex.append_journal(directory / 'api/requests.jsonl', dict(query_id=qid, request=ex.request_for(plan, q),
                       transport='batch', batch_job_key=job['job_key']))
    batch.submit(client, plan, directory, state)
    client.finish()
    batch.collect(client, plan, directory, state)
    state['jobs'][0]['collected'] = False  # Crash after append, before saving collection state.
    before = (directory / 'api/responses.jsonl').read_bytes()
    batch.collect(client, plan, directory, state)
    assert (directory / 'api/responses.jsonl').read_bytes() == before


def test_modified_input_and_new_sync_attempt_are_rejected(run):
    plan, directory = run
    state = batch.prepare(plan, directory)
    q = plan['queries'][1]
    ex.append_journal(directory / 'api/requests.jsonl', dict(query_id=q['query_id'], request=ex.request_for(plan, q)))
    with pytest.raises(ValueError, match='another runner'):
        batch.load_state(plan, directory)
    path = directory / 'api/batch' / state['jobs'][0]['input_file']
    path.write_text('edited')
    with pytest.raises(ValueError, match='hash/path'):
        batch.load_state(plan, directory)



def test_execute_submit_collect_resume_and_no_client_when_done(run):
    plan, directory = run
    client = FakeClient()
    batch.execute(plan, directory, repository_root=HERE.parents[1], client_factory=lambda: client)
    assert len(client.created) == 3 and client.closed
    batch.execute(plan, directory, repository_root=HERE.parents[1], client_factory=lambda: client)
    assert len(client.created) == 3
    client.finish()
    summary = batch.execute(plan, directory, repository_root=HERE.parents[1],
                            submit_new=False, client_factory=lambda: client)
    assert summary['responses_saved'] == 10
    batch.execute(plan, directory, repository_root=HERE.parents[1], submit_new=False,
                  client_factory=lambda: pytest.fail('Completed runs need no API'))
    assert len(client.created) == 3


def test_collect_never_submits_prepared_shards(run):
    plan, directory = run
    state = batch.prepare(plan, directory, max_requests=2)
    client = FakeClient()
    batch.submit(client, plan, directory, state)
    client.finish()
    batch.execute(plan, directory, repository_root=HERE.parents[1], submit_new=False,
                  wait=True, client_factory=lambda: client)
    assert len(client.created) == 3
    assert any(j['status'] == 'prepared' for j in batch.load_state(plan, directory)['jobs'])


def test_wait_collects_then_scores(run, monkeypatch):
    plan, directory = run
    client = FakeClient()
    monkeypatch.setattr(batch.time, 'sleep', lambda seconds: client.finish())
    result = batch.execute(plan, directory, repository_root=HERE.parents[1], wait=True,
                           client_factory=lambda: client)
    assert result['status'] == 'complete'
    assert len(client.created) == 3


def test_terminal_failure_stops_new_submissions(run):
    plan, directory = run
    state = batch.prepare(plan, directory, max_requests=2)
    client = FakeClient()
    batch.submit(client, plan, directory, state)
    client.finish(fail=True)
    with pytest.raises(RuntimeError, match='failures require review'):
        batch.execute(plan, directory, repository_root=HERE.parents[1], client_factory=lambda: client)
    assert len(client.created) == 3
    assert (directory / 'analysis' / evaluation.SCORING_VERSION / 'per_query.json').exists()


def test_batch_source_integrity_and_sync_guard(run):
    plan, directory = run
    plan['source_sha256'] = {'pyproject.toml': 'wrong'}
    plan.pop('plan_id')
    plan['plan_id'] = ex.digest(plan)
    ex.write_json(directory / 'manifest.json', plan)
    with pytest.raises(ValueError, match='Source changed'):
        batch.execute(plan, directory, repository_root=HERE.parents[1],
                      client_factory=lambda: pytest.fail('API must not start'))
    plan['source_sha256'] = {}
    plan.pop('plan_id')
    plan['plan_id'] = ex.digest(plan)
    ex.write_json(directory / 'manifest.json', plan)
    batch.prepare(plan, directory)
    with pytest.raises(ValueError, match='synchronous runner'):
        ex.execute(plan, directory, repository_root=HERE.parents[1], resume=True,
                   client_factory=lambda: pytest.fail('Sync must not start'))


def test_cli_execute_routes_to_batch(run, monkeypatch):
    from trajectory_experiment import runner
    from trajectory_experiment.config import Experiment
    plan, directory = run
    monkeypatch.setattr(runner, 'HERE', directory)
    monkeypatch.setattr(runner, 'make_plan', lambda experiment: plan)
    monkeypatch.setattr(ex, 'save_plan', lambda *args: None)
    monkeypatch.setattr(ex, 'execute', lambda *a, **k: pytest.fail('Sync runner called'))
    calls = []
    monkeypatch.setattr(batch, 'execute', lambda *a, **k: calls.append(k))
    assert runner.run(Experiment(), ['--execute']) == 0
    assert len(calls) == 1 and not calls[0]['wait']
    assert runner.run(Experiment(), ['--execute', '--wait', '--resume']) == 0
    assert calls[1]['wait']


def test_collect_before_submission_requires_no_client(run):
    plan, directory = run
    batch.prepare(plan, directory)
    with pytest.raises(ValueError, match='use --execute first'):
        batch.execute(plan, directory, repository_root=HERE.parents[1], submit_new=False,
                      client_factory=lambda: pytest.fail('No network for an unsubmitted run'))
