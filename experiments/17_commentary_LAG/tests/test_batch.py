from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent/'13_LAG')]
from commentary_experiment import batch, execution as ex, evaluation
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
                              body={'choices': [{'message': {'content': '1. Scene overview: recorded game commentary'}}],
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


def test_submit_collect_idempotent(run):
    plan, directory = run
    client = FakeClient()
    batch.execute(plan, directory, repository_root=HERE.parents[1], client_factory=lambda: client)
    assert len(client.created) == 3
    client.finish()
    result = batch.execute(plan, directory, repository_root=HERE.parents[1], submit_new=False, client_factory=lambda: client)
    assert result['responses_saved'] == 10
    before = (directory/'api/responses.jsonl').read_bytes()
    batch.execute(plan, directory, repository_root=HERE.parents[1], client_factory=lambda: client)
    assert len(client.created) == 3
    assert (directory/'api/responses.jsonl').read_bytes() == before
    exported = ex.read_json(directory/'evaluation/responses.json')
    assert len(exported) == 10 and all(x['status'] == 'received' for x in exported)


def test_lost_submission_recovered(run):
    plan, directory = run
    client = FakeClient()
    client.fail_create = True
    with pytest.raises(TimeoutError):
        batch.execute(plan, directory, repository_root=HERE.parents[1], client_factory=lambda: client)
    assert len(client.created) == 1
    client.fail_create = False
    batch.execute(plan, directory, repository_root=HERE.parents[1], client_factory=lambda: client)
    assert len(client.created) == 3


def test_partial_failure_preserves_answers(run):
    plan, directory = run
    client = FakeClient()
    batch.execute(plan, directory, repository_root=HERE.parents[1], client_factory=lambda: client)
    client.finish(fail=True)
    result = batch.execute(plan, directory, repository_root=HERE.parents[1], submit_new=False, client_factory=lambda: client)
    assert result['responses_saved'] == 7 and result['unresolved'] == 3
    exported = ex.read_json(directory/'evaluation/responses.json')
    assert sum(x['status'] == 'error' for x in exported) == 3
