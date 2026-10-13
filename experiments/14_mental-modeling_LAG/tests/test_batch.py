"""Batch migration contracts; all API objects are local fakes."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import batch_run as batch
from lag_experiment import execution as ex, evaluation, prompts


def obj(data):
    return SimpleNamespace(**data, model_dump=lambda **kwargs: deepcopy(data))


@pytest.fixture
def run(tmp_path):
    sample = dict(messages=[dict(role='user', content='Predict an action')],
                  episode_id='episode', target_id='A0100', task='next-action', H=2,
                  start=0, query_step=2, target_step=2, sample_number=0, problem_id='problem',
                  ground_truth=[1, 2, 0], prediction_fields=['a', 'b', 'c'])
    plan = dict(samples={'sample': sample}, queries=[], request_options={'reasoning_effort': 'medium'}, source_sha256={})
    for i in range(10):
        alias = ('sol', 'terra', 'luna')[i % 3]
        q = dict(sample_id='sample', model=f'gpt-5.6-{alias}', model_alias=alias,
                 repeat_id=i, condition_id=alias)
        q['request_sha256'] = ex.digest(ex.request_for(plan, q))
        q['query_id'] = ex.digest(q)
        plan['queries'].append(q)
    plan['plan_id'] = ex.digest(plan)
    ex.write_json(tmp_path / 'manifest.json', plan)
    ex.write_json(tmp_path / 'api/identity.json', {'plan_id': plan['plan_id']})
    q = plan['queries'][0]
    ex.append_journal(tmp_path / 'api/requests.jsonl', dict(query_id=q['query_id'], request=ex.request_for(plan, q)))
    ex.append_journal(tmp_path / 'api/responses.jsonl', dict(query_id=q['query_id'], raw_response={'choices': []}))
    return plan, tmp_path


class FakeClient:
    def __init__(self):
        self.inputs, self.outputs, self.remote, self.created = {}, {}, {}, []
        self.fail_create = False
        self.files = SimpleNamespace(create=self.upload, content=lambda id: SimpleNamespace(text=self.outputs[id]))
        self.batches = SimpleNamespace(create=self.create, retrieve=lambda id: obj(self.remote[id]),
                                       list=lambda **kw: [obj(x) for x in self.remote.values()])

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
                              body={'choices': [{'message': {'content': prompts.ACTION_MARKER + ' [1,2,0]'}}],
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


def test_prepare_partition_excludes_sync_and_preserves_inputs(run):
    plan, directory = run
    before = (directory / 'api/requests.jsonl').read_bytes()
    state = batch.prepare(plan, directory, after_sync=1, max_requests=2)
    assert sum(len(j['query_ids']) for j in state['jobs']) == 9
    assert len(state['jobs']) == 6
    assert all(len(j['query_ids']) <= 2 for j in state['jobs'])
    assert (directory / 'api/requests.jsonl').read_bytes() == before
    assert batch.load_state(plan, directory) == state
    assert batch.validate_plan(directory) == plan


def test_prepare_requires_resolved_exact_prefix(run):
    plan, directory = run
    with pytest.raises(ValueError, match='exactly the first'):
        batch.prepare(plan, directory, after_sync=2)
    (directory / 'api/responses.jsonl').write_text('')
    with pytest.raises(ValueError, match='saved responses'):
        batch.prepare(plan, directory, after_sync=1)


def test_full_lifecycle_out_of_order_and_idempotent_collection(run):
    plan, directory = run
    state = batch.prepare(plan, directory, after_sync=1, max_requests=2)
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
    assert sum(r['parse_success'] for r in records) == 9
    assert all(r['elapsed_seconds'] is None for r in records)


def test_lost_create_response_recovers_without_resubmit(run):
    plan, directory = run
    state = batch.prepare(plan, directory, after_sync=1)
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
    state = batch.prepare(plan, directory, after_sync=1)
    client = FakeClient()
    state['jobs'][0].update(status='submitting', file_id='file_unknown')
    with pytest.raises(RuntimeError, match='No automatic resend'):
        batch.collect(client, plan, directory, state)
    assert client.created == []


@pytest.mark.parametrize('omit', [False, True])
def test_expired_partial_results_saved_and_not_resent(run, omit):
    plan, directory = run
    state = batch.prepare(plan, directory, after_sync=1)
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
    state = batch.prepare(plan, directory, after_sync=1)
    client = FakeClient()
    batch.submit(client, plan, directory, state)
    client.finish()
    key = 'batch_0_out'
    client.outputs[key] += '\n' + client.outputs[key].splitlines()[0]
    with pytest.raises(ValueError, match='duplicate custom_id'):
        batch.collect(client, plan, directory, state)
    assert len(ex.read_journal(directory / 'api/responses.jsonl')) == 1


def test_partial_reservation_and_partial_import_can_resume(run):
    plan, directory = run
    state = batch.prepare(plan, directory, after_sync=1)
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
    state = batch.prepare(plan, directory, after_sync=1)
    q = plan['queries'][1]
    ex.append_journal(directory / 'api/requests.jsonl', dict(query_id=q['query_id'], request=ex.request_for(plan, q)))
    with pytest.raises(ValueError, match='another runner'):
        batch.load_state(plan, directory)
    path = directory / 'api/batch' / state['jobs'][0]['input_file']
    path.write_text('edited')
    with pytest.raises(ValueError, match='hash/path'):
        batch.load_state(plan, directory)


def test_completion_scope_excludes_untouched_conditions_and_stays_bounded(run):
    plan, directory = run
    # The prefix touches H2 only; later H5 queries must never be submitted.
    plan['samples']['later'] = dict(plan['samples']['sample'], H=5)
    for i, q in enumerate(plan['queries']):
        if i >= 6:
            q['sample_id'] = 'later'
            q['request_sha256'] = ex.digest(ex.request_for(plan, q))
            q.pop('query_id')
            q['query_id'] = ex.digest(q)
    plan.pop('plan_id')
    plan['plan_id'] = ex.digest(plan)
    ex.write_json(directory / 'manifest.json', plan)
    ex.write_json(directory / 'api/identity.json', {'plan_id': plan['plan_id']})
    state = batch.prepare(plan, directory, after_sync=1, completion_only=True)
    assert state['scope'] == 'prefix-completion'
    assert sum(len(j['query_ids']) for j in state['jobs']) == 5
    assert batch.load_state(plan, directory) == state
    client = FakeClient()
    batch.submit(client, plan, directory, state)
    assert len(client.created) == 3
    assert sum(len(rows) for rows in client.inputs.values()) == 5
    client.finish()
    batch.collect(client, plan, directory, state)
    batch.submit(client, plan, directory, state)
    assert len(client.created) == 3  # Rerunning never advances to untouched H5.
    assert len(ex.read_journal(directory / 'api/requests.jsonl')) == 6
    with pytest.raises(ValueError, match='scope differs'):
        batch.prepare(plan, directory, after_sync=1)


def test_real_prefix_completion_is_exactly_five():
    plan = ex.read_json(HERE / 'results/sliding_window_v1/manifest.json')
    selected = batch.completion_query_ids(plan, 220)
    queries = [q for q in plan['queries'] if q['query_id'] in selected]
    assert len(queries) == 5
    from collections import Counter
    assert Counter(q['model_alias'] for q in queries) == {'sol': 1, 'terra': 2, 'luna': 2}
    assert all(q['condition_id'].endswith('/A0100/H5/last-state/self') for q in queries)
