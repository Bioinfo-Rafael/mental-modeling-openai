import copy
import importlib
import json
import threading

import pytest

w = importlib.import_module('experiments.19_RealTime_Lag_v2.realtime_experiment.workflow')
replay = importlib.import_module('experiments.19_RealTime_Lag_v2.realtime_experiment.replay')


@pytest.fixture
def plan(tmp_path, monkeypatch):
    monkeypatch.setattr(w, 'RESULTS', tmp_path / 'results')
    monkeypatch.setattr(w, 'VIEWER', tmp_path / 'viewer')
    return w.create_plan('test', 1, 1)


class Client:
    api_key = 'test-key-never-sent'

    def __init__(self, barrier=False):
        self.calls = []
        self.barrier = threading.Barrier(2) if barrier else None

    def post(self, path, cast_to, body):
        self.calls.append(copy.deepcopy(body))
        question = body['questions'][0]
        choices = question['choices']
        if question['name'] == 'changed_aircraft':
            choice = 'A1'
        else:
            if self.barrier:
                self.barrier.wait(timeout=2)
            choice = choices[0]['value']
        return {'answers': [{'name': question['name'], 'type': 'choice', 'choice': choice,
            'confidence': 0.9, 'probabilities': [
                {'value': c['value'], 'probability': float(c['value'] == choice)} for c in choices]}],
            'usage': {'input_tokens': 100, 'output_tokens': 0}}


def test_frozen_causal_requests_and_counts(plan):
    m = w.verify(plan)
    assert len(m['queries']) == 7
    assert m['samples'][0]['end_step'] == 39
    window = w.read_json(plan / 'inputs/end_0039.json')
    assert [b['step'] for b in window['blocks']] == list(range(40))
    for q in m['queries']:
        body = w.request_for(plan, q)
        assert len(body['questions'][0]['choices']) == (q['choice_count'] or 4)
        if q['stage'] == 'goal':
            assert q['aircraft_id'] in body['input'][1]['content']
            assert '61 consecutive' not in body['input'][0]['content']
            assert 'DETECTION TARGET' not in body['input'][0]['content']
    (plan / 'choices/4.json').write_text('[]')
    with pytest.raises(ValueError, match='Frozen file changed'):
        w.verify(plan)


def test_a1_gate_sends_both_aircraft_in_parallel_at_all_counts(plan):
    client = Client(barrier=True)
    w.execute(plan, client=client)
    assert len(client.calls) == 7
    records = w.records_for(plan)
    assert len(records) == 7 and all(r['api_status'] == 'ok' for r in records.values())
    goals = [b for b in client.calls if b['questions'][0]['name'] == 'primary_goal']
    assert sum('Target aircraft: A0100' in b['input'][1]['content'] for b in goals) == 3
    assert sum('Target aircraft: A0200' in b['input'][1]['content'] for b in goals) == 3
    w.execute(plan, resume=True, client=client)
    assert len(client.calls) == 7  # complete run does not resend


def test_real_sdk_transport_and_response_parsing(plan):
    import httpx
    from openai import OpenAI

    fake = Client()

    def respond(request):
        assert request.method == 'POST' and request.url.path == '/v1/decisions'
        return httpx.Response(200, json=fake.post('/decisions', None, json.loads(request.content)))

    with OpenAI(api_key='test-only', max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
        w.execute(plan, client=client)
    assert len(fake.calls) == 7
    assert all(r['api_status'] == 'ok' for r in w.records_for(plan).values())
    assert w.export(plan)['complete']


@pytest.mark.parametrize('choice,latency,status,expected', [
    ('A1',2.999,'ok',True),('A2',3.0,'ok',True),('BOTH',3.001,'ok',False),
    ('NONE',0.1,'ok',False),('A1',0.1,'invalid_output',False)])
def test_gate_deadlines(choice,latency,status,expected):
    assert w.gate_eligible({'choice':choice,'latency_s':latency,'api_status':status}) is expected


def test_late_gate_never_starts_goals(plan):
    gate = w.verify(plan)['queries'][0]
    w.write_json(plan / 'responses' / (gate['id']+'.json'), {
        'query_id':gate['id'],'api_status':'ok','choice':'BOTH','latency_s':3.001})
    client = Client()
    w.execute(plan, resume=True, client=client)
    assert not client.calls
    assert sum(r['api_status']=='skipped_gate' for r in w.records_for(plan).values()) == 6
    result = w.read_json(plan / 'replay.json')['events'][0]
    assert result['status'] == 'late' and result['choice'] is None
    assert result['event_sim_s'] == pytest.approx(10.8)


def test_interrupted_started_query_is_not_resent(plan):
    gate = w.verify(plan)['queries'][0]
    w.storage.append_jsonl(plan / 'requests.jsonl', {'query_id':gate['id']})
    client = Client()
    w.execute(plan, resume=True, client=client)
    assert not client.calls
    assert w.records_for(plan)[gate['id']]['api_status'] == 'uncertain'


def test_limit_resume_preserves_parallel_pairs(plan):
    client = Client()
    w.execute(plan, limit=1, client=client)
    assert len(client.calls)==1
    w.execute(plan, resume=True, limit=1, client=client)
    assert len(client.calls)==1
    w.execute(plan, resume=True, client=client)
    assert len(client.calls)==7


def test_api_error_stops_and_redacts_key(plan):
    class Failing(Client):
        def post(self, *args, **kwargs):
            raise RuntimeError('invalid '+self.api_key)
    w.execute(plan, client=Failing())
    records=w.records_for(plan)
    assert len(records)==1
    assert next(iter(records.values()))['error']=='invalid [REDACTED]'


def test_replay_independent_deadlines_and_stale_order():
    queries=[]; records={}
    def add(end, goal_latency, choice, gate_latency=1.0, aircraft='A0100'):
        gate_id=f'g{end}'
        if gate_id not in records:
            queries.append({'id':gate_id,'stage':'gate','repeat':1,'choice_count':None,'aircraft_id':None,'source_end_step':end})
            records[gate_id]={'api_status':'ok','choice':'A1','latency_s':gate_latency}
        qid=f'p{end}_{aircraft}'
        queries.append({'id':qid,'stage':'goal','repeat':1,'choice_count':4,'aircraft_id':aircraft,'source_end_step':end,'gate_query_id':gate_id})
        records[qid]={'api_status':'ok','choice':choice,'latency_s':goal_latency,'dispatch_offset_s':0}
    add(50,2.5,'A',gate_latency=2.0)  # total 4.5s, valid but arrives after newer answer
    add(55,0.1,'B')
    add(60,0.1,'B')  # same label still advances latest source step
    add(65,0.1,'A')
    add(65,3.001,'C',aircraft='A0200')
    m={'run_id':'test','created_at':'test','repeats':1,'recording_last_step':70,'queries':queries,
       'config':{'episode_id':'test','dt_s':.2,'deadline':{'limit_seconds':3.0}}}
    cat={'levels':{'4':[{'value':v,'label':v} for v in 'ABCD'],'8':[],'32':[]}}
    result=replay.build_replay(m,records,cat)
    byid={e['id']:e for e in result['events']}
    assert byid['p50_A0100']['ready_sim_s']==14.5
    assert byid['p50_A0100']['status']=='stale'
    assert byid['p55_A0100']['status']=='displayed'
    assert byid['p60_A0100']['status']=='same_goal'
    assert byid['p65_A0100']['status']=='displayed'
    assert byid['p65_A0200']['status']=='late'
    assert byid['p65_A0200']['event_sim_s']==17
    assert byid['p65_A0200']['choice'] is None


def test_probability_vector_validation():
    body={'questions':[{'name':'primary_goal','choices':[{'value':'A'},{'value':'B'}]}]}
    raw={'answers':[{'name':'primary_goal','type':'choice','choice':'A','confidence':.9,
                    'probabilities':[{'value':'A','probability':.5},{'value':'B','probability':.4}]}]}
    with pytest.raises(ValueError,match='sum'):
        w.parse_answer(raw,body)
