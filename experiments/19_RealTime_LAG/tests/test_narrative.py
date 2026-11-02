import importlib
import json

import httpx
from openai import OpenAI
import pytest

n = importlib.import_module('experiments.19_RealTime_LAG.realtime_experiment.narrative')
w = n.w


@pytest.fixture
def plan(tmp_path, monkeypatch):
    monkeypatch.setattr(w, 'RESULTS', tmp_path / 'results')
    monkeypatch.setattr(w, 'VIEWER', tmp_path / 'viewer')
    source = w.create_plan('source', 1, 2)
    for q in w.verify(source)['queries']:
        if q['stage'] == 'gate':
            w.write_json(source / 'responses' / f"{q['id']}.json", {
                'query_id': q['id'], 'api_status': 'ok', 'latency_s': .5,
                'choice': 'A1' if q['source_end_step'] == 39 else 'NONE'})
    return n.create_plan('narrative_test', 'source')


def response(content=None):
    return {'id': 'test', 'object': 'chat.completion', 'created': 0, 'model': n.MODEL,
            'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {
                'role': 'assistant', 'content': json.dumps(content or {'A0100': 'First purpose.', 'A0200': 'Second purpose.'})}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 10, 'total_tokens': 20}}


def test_frozen_prompt_two_aircraft_causal_window_and_positive_filter(plan):
    m = w.verify(plan)
    assert len(m['queries']) == 1
    assert m['queries'][0]['gate_choice'] == 'A1'
    body = n.request_for(plan, m['queries'][0])
    assert body['model'] == 'gpt-5.6-sol' and body['reasoning_effort'] == 'medium'
    system, user = [v['content'] for v in body['messages']]
    assert '61' not in system and 'exactly two string fields' in system
    assert 'Analyze all four aircraft' in system  # opponents still provide evidence
    assert '35–50 English words' in system
    assert 'Current step t: 39' in user and 'steps 0 through 39' in user
    window = w.read_json(plan / 'inputs/end_0039.json')
    assert max(b['step'] for b in window['blocks']) == 39
    (plan / 'prompts/system.txt').write_text(system + 'changed')
    with pytest.raises(ValueError, match='Frozen file changed'):
        w.verify(plan)


def test_actual_sdk_measures_both_answers_and_resume_never_resends(plan, monkeypatch):
    calls = []
    ticks = iter([100., 112.5])
    # Isolate the timer module so httpx/OpenAI keep their normal clocks.
    from types import SimpleNamespace
    monkeypatch.setattr(n, 'time', SimpleNamespace(perf_counter=lambda: next(ticks)))
    def transport(request):
        calls.append(json.loads(request.content))
        assert request.url.path == '/v1/chat/completions'
        return httpx.Response(200, json=response())
    with OpenAI(api_key='test-only', max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(transport))) as client:
        n.execute(plan, client=client)
        n.execute(plan, resume=True, client=client)
    assert len(calls) == 1
    records = list(w.records_for(plan).values())
    assert records[0]['latency_s'] == 12.5 and records[0]['api_status'] == 'ok'
    replay = w.read_json(plan / 'replay.json')
    assert len(replay['events']) == 2
    assert all(e['event_sim_s'] == pytest.approx(7.8) and e['latency_s'] == 12.5 for e in replay['events'])
    assert replay['last_step'] == w.verify(plan)['recording_last_step']
    assert replay['summary']['all_successful']


def test_invalid_four_aircraft_output_is_preserved(plan):
    def transport(request):
        return httpx.Response(200, json=response({'A0100': 'a', 'A0200': 'b', 'B0100': 'c', 'B0200': 'd'}))
    with OpenAI(api_key='test-only', max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(transport))) as client:
        n.execute(plan, client=client)
    r = next(iter(w.records_for(plan).values()))
    assert r['api_status'] == 'invalid_output' and 'B0100' in r['raw']['choices'][0]['message']['content']
    replay = w.read_json(plan / 'replay.json')
    assert not replay['events'] and not replay['summary']['all_successful']


def test_uncertain_started_request_is_not_resent(plan):
    q = w.verify(plan)['queries'][0]
    w.storage.append_jsonl(plan / 'requests.jsonl', {'query_id': q['id']})
    n.execute(plan, resume=True, client=object())
    assert w.records_for(plan)[q['id']]['api_status'] == 'uncertain'


def test_http_failure_stops_and_does_not_retry(plan):
    calls = []
    def transport(request):
        calls.append(request.url.path)
        return httpx.Response(429, json={'error': {'message': 'rate limit', 'type': 'rate_limit_error'}})
    with OpenAI(api_key='test-only', max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(transport))) as client:
        n.execute(plan, client=client)
    assert len(calls) == 1
    assert next(iter(w.records_for(plan).values()))['api_status'] == 'error'


def test_explicit_retry_preserves_failed_attempt_and_skips_success(plan):
    calls = []
    def transport(request):
        calls.append(request.url.path)
        if len(calls) == 1:
            raise httpx.ConnectError('test connection failure', request=request)
        return httpx.Response(200, json=response())
    with OpenAI(api_key='test-only', max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(transport))) as client:
        n.execute(plan, client=client)
        first = next(iter(w.records_for(plan).values()))
        assert first['api_status'] == 'error' and first['cause_types'] == ['ConnectError']
        n.execute(plan, resume=True, client=client)
        assert len(calls) == 1  # ordinary resume does not retry a recorded failure
        n.execute(plan, resume=True, retry_failed=True, client=client)
        n.execute(plan, resume=True, retry_failed=True, client=client)
    assert len(calls) == 2
    latest = next(iter(w.records_for(plan).values()))
    assert latest['api_status'] == 'ok' and latest['attempt'] == 2
    archive = plan / 'attempts' / latest['query_id']
    assert w.read_json(archive / 'attempt_001.json') == first
    assert w.read_json(archive / 'attempt_002.json') == latest
    assert len(w.storage.read_jsonl(plan / 'responses.jsonl')) == 2


def test_interrupted_retry_is_uncertain_and_not_sent_again(plan):
    q = w.verify(plan)['queries'][0]
    old = {'query_id': q['id'], 'api_status': 'error', 'latency_s': 44.0}
    w.write_json(plan / 'responses' / f"{q['id']}.json", old)
    w.write_json(plan / 'attempts' / q['id'] / 'attempt_001.json', old)
    for attempt in (1, 2):
        w.storage.append_jsonl(plan / 'requests.jsonl', {'query_id': q['id'], 'attempt': attempt})
    n.execute(plan, resume=True, retry_failed=True, client=object())
    n.execute(plan, resume=True, retry_failed=True, client=object())
    assert w.records_for(plan)[q['id']]['api_status'] == 'uncertain'
    assert w.records_for(plan)[q['id']]['attempt'] == 2
    assert w.read_json(plan / 'attempts' / q['id'] / 'attempt_001.json') == old
