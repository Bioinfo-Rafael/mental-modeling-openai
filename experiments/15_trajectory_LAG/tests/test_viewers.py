"""Data-integrity checks for the offline timeline and raw-response viewers."""
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import build_results_viewer as raw_view
import plot_trajectory_timelines as timeline


def test_missing_dead_and_invalid_telemetry_stays_missing():
    base = dict(alive=1, state_valid=1, action_status='recorded',
                **{field: float(i) for i, field in enumerate(timeline.STATE_FEATURES)},
                **{f'{field}_id': 1 for field in timeline.ACTIONS})
    records = {(0, 'A'): base, (1, 'A'): dict(base, alive=0),
               (2, 'A'): dict(base, state_valid=0), (3, 'A'): dict(base, action_status='missing')}
    window = timeline.recorded_window(records, 'A', -1, 3)
    assert window[0]['state'] == window[2]['state'] == window[3]['state'] == [None]*10
    assert window[1]['state'] == list(range(10))
    assert window[3]['action'] == [1]*4
    assert window[4]['state'] == list(range(10)) and window[4]['action'] == [None]*4


def test_angular_wrap_keeps_both_recorded_endpoints():
    x, y = timeline.line_points([20, 21, 22], [359, 1, 2], circular=True)
    assert x[0] == 20 and y[0] == 359
    assert math.isnan(x[1]) and math.isnan(y[1])
    assert x[2:] == [21, 22] and y[2:] == [1, 2]


def test_snapshot_excludes_only_uncommitted_trailing_bytes(tmp_path):
    path = tmp_path / 'journal.jsonl'
    path.write_bytes(b'{"query_id":"a"}\n{"query_id":')
    rows, meta = raw_view.journal_snapshot(path)
    assert rows == [{'query_id': 'a'}] and meta['trailing_bytes_excluded'] > 0
    path.write_bytes(b'not json\n')
    with pytest.raises(json.JSONDecodeError):
        raw_view.journal_snapshot(path)


def test_embedded_json_cannot_close_script():
    rendered = raw_view.render_html({'response': '</script><img src=x>', 'x': '\u2028'}, '__DATASET__')
    assert '</script>' not in rendered
    assert json.loads(rendered)['response'] == '</script><img src=x>'


@pytest.fixture
def completed_run(tmp_path):
    truth = [dict(step=t, state=[float(t)] * 10, action=[1, 2, 0]) for t in range(20, 30)]
    fields = dict(state=list(timeline.STATE_FEATURES), action=list(timeline.ACTIONS[:3]))
    sample = dict(episode_id='episode', target_id='A0100', H=20, task='trajectory',
                  observation_scope='self', sample_number=0, start=0, target_step=20,
                  target_steps=list(range(20, 30)), horizon=10, problem_id='problem',
                  ground_truth=truth, prediction_fields=fields, messages=[dict(role='user', content='forecast')])
    plan = dict(samples={'sid': sample}, queries=[], request_options={})
    for alias in ('sol', 'terra', 'luna'):
        q = dict(sample_id='sid', model_alias=alias, model=alias, condition_id=alias)
        q['request_sha256'] = raw_view.ex.digest(raw_view.ex.request_for(plan, q))
        q['query_id'] = raw_view.ex.digest(q)
        plan['queries'].append(q)
    plan['plan_id'] = raw_view.ex.digest(plan)
    raw_view.ex.write_json(tmp_path / 'manifest.json', plan)
    raw_view.ex.write_json(tmp_path / 'api/identity.json', dict(plan_id=plan['plan_id']))
    for q in plan['queries']:
        raw_view.ex.append_journal(tmp_path / 'api/requests.jsonl', dict(query_id=q['query_id'], request=raw_view.ex.request_for(plan, q)))
    for q in reversed(plan['queries']):
        prediction = deepcopy(truth)
        if q['model_alias'] == 'terra':
            prediction[0]['state'][2] += 7
        text = '>>Final trajectory: ' + json.dumps(prediction)
        if q['model_alias'] == 'luna':
            text = 'invalid response'
        raw_view.ex.append_journal(tmp_path / 'api/responses.jsonl', dict(query_id=q['query_id'],
            raw_response=dict(choices=[dict(message=dict(content=text))])))
    return plan, tmp_path


def test_reversed_response_order_keeps_model_and_truth_alignment(completed_run):
    plan, source = completed_run
    predictions = timeline.predictions_from_saved(plan, source)['sid']
    assert predictions['sol']['prediction'] == plan['samples']['sid']['ground_truth']
    assert predictions['terra']['prediction'][0]['state'][2] == 27
    assert predictions['luna']['status'] == 'parse_error' and predictions['luna']['prediction'] is None
    data = raw_view.load_data(source)
    assert data['counts'] == dict(trajectory_scored=2, parse_error=1)
    assert len({row['comparison_id'] for row in data['rows']}) == 1
    assert data['rows'][1]['scoring']['state_mae_by_field']['alt_m'] == .7
    assert data['rows'][2]['raw_response']['choices'][0]['message']['content'] == 'invalid response'


def test_duplicate_unknown_and_mismatched_identity_rejected(completed_run):
    plan, source = completed_run
    path = source / 'api/responses.jsonl'
    original = path.read_text()
    path.write_text(original + original.splitlines()[0] + '\n')
    for load in (lambda: timeline.predictions_from_saved(plan, source), lambda: raw_view.load_data(source)):
        with pytest.raises(ValueError, match='[Dd]uplicate'):
            load()
    path.write_text(original)
    raw_view.ex.write_json(source / 'api/identity.json', {'plan_id': 'wrong'})
    for load in (lambda: timeline.predictions_from_saved(plan, source), lambda: raw_view.load_data(source)):
        with pytest.raises(ValueError, match='identity differs'):
            load()
