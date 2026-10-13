"""Offline coverage of rolling slots, priority, migration, budgets and restart."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import batch_run as batch
import batch_scheduler as rolling
from lag_experiment import execution as ex, prompts
from test_batch import FakeClient


@pytest.fixture
def grid(tmp_path):
    plan = dict(experiment=dict(episodes=['match'], target_ids=['A0100', 'A0200'],
                               histories=[2, 5], tasks=['next-action', 'last-action', 'next-state', 'last-state']),
                samples={}, queries=[], request_options={'reasoning_effort': 'medium'}, source_sha256={})
    for target in plan['experiment']['target_ids']:
        for h in plan['experiment']['histories']:
            for task in plan['experiment']['tasks']:
                for scope in ('self', 'all_aircraft_missiles'):
                    for n in range(5):
                        s = dict(episode_id='match', target_id=target, H=h, task=task, observation_scope=scope,
                                 sample_number=n, start=n, target_step=n+h, query_step=n+h,
                                 problem_id=f'{target}/{h}/{task}/{n}', ground_truth=[1, 2, 0],
                                 prediction_fields=['a', 'b', 'c'],
                                 messages=[dict(role='user', content=f'{target} {h} {task} {scope} {n}')])
                        sid = ex.digest(s)[:24]
                        plan['samples'][sid] = s
                        for alias in ('sol', 'terra', 'luna'):
                            q = dict(sample_id=sid, model='gpt-5.6-'+alias, model_alias=alias, repeat_id=1,
                                     condition_id=f'{alias}/match/{target}/{h}/{task}/{scope}')
                            q['request_sha256'] = ex.digest(ex.request_for(plan, q))
                            q['query_id'] = ex.digest(q)
                            plan['queries'].append(q)
    plan['plan_id'] = ex.digest(plan)
    ex.write_json(tmp_path / 'manifest.json', plan)
    ex.write_json(tmp_path / 'api/identity.json', {'plan_id': plan['plan_id']})
    for q in plan['queries'][:10]:
        ex.append_journal(tmp_path / 'api/requests.jsonl', dict(query_id=q['query_id'], request=ex.request_for(plan, q)))
        ex.append_journal(tmp_path / 'api/responses.jsonl', dict(query_id=q['query_id'],
                          raw_response={'choices': [{'message': {'content': prompts.ACTION_MARKER+' [1,2,0]'}}]}))
    return plan, tmp_path


def finish_only(client, ids):
    unchanged = {k: deepcopy(v) for k, v in client.remote.items() if k not in ids}
    client.finish()
    client.remote.update(unchanged)


def started(grid):
    plan, directory = grid
    old = batch.prepare(plan, directory, after_sync=10, completion_only=True)
    client = FakeClient()
    batch.submit(client, plan, directory, old)
    state = rolling.prepare_rolling(plan, directory, after_sync=10)
    return plan, directory, state, client, old


def active_sets(state):
    jobs = {j['name']: j for j in state['jobs']}
    return [s for s in state['sets'] if s['admitted'] and not rolling.done(s, jobs)]


def job_ids(state, group):
    return [j['batch_id'] for j in state['jobs'] if j['name'] in group['jobs'] and not j.get('collected')]


def test_migration_preserves_submitted_jobs_and_prefix_then_pairs_scopes(grid):
    plan, directory, state, client, old = started(grid)
    assert ex.read_json(directory / 'api/batch/state.before_rolling.json') == old
    assert state['scope'] == 'rolling-comparison'
    assert len(state['sets']) == 16
    assert all(len(s['query_ids']) == 30 for s in state['sets'])
    assert [s['condition'] for s in state['sets'][:5]] == [
        dict(episode_id='match', target_id='A0100', H=h, task=t)
        for h, t in [(2, 'next-action'), (2, 'last-action'), (2, 'next-state'), (2, 'last-state'), (5, 'next-action')]]
    for before in old['jobs']:
        after = next(j for j in state['jobs'] if j['name'] == before['name'])
        assert all(after[k] == v for k, v in before.items())
    before = (directory / 'api/requests.jsonl').read_bytes()
    assert rolling.prepare_rolling(plan, directory) == state
    assert (directory / 'api/requests.jsonl').read_bytes() == before
    assert rolling.submit_ready(client, plan, directory, state) == 0  # Bootstrap still running.
    assert len(client.created) == 3
    client.finish()
    batch.collect(client, plan, directory, state)
    assert rolling.submit_ready(client, plan, directory, state) == 12
    assert len(active_sets(state)) == 4
    assert len(ex.read_journal(directory / 'api/requests.jsonl')) == 120
    assert sum(j['status'] != 'prepared' and not j.get('collected') for j in state['jobs']) == 12


def test_refill_after_one_complete_set_not_after_one_model_or_entire_wave(grid):
    plan, directory, state, client, _ = started(grid)
    client.finish()
    batch.collect(client, plan, directory, state)
    rolling.submit_ready(client, plan, directory, state)
    group = state['sets'][1]
    ids = job_ids(state, group)
    finish_only(client, ids[:1])
    batch.collect(client, plan, directory, state)
    assert rolling.submit_ready(client, plan, directory, state) == 0
    finish_only(client, ids[1:])
    batch.collect(client, plan, directory, state)
    assert rolling.submit_ready(client, plan, directory, state) == 3
    assert [s['set_id'] for s in active_sets(state)] == [state['sets'][i]['set_id'] for i in (0, 2, 3, 4)]
    reloaded = batch.load_state(plan, directory)
    assert rolling.submit_ready(client, plan, directory, reloaded) == 0
    assert len(client.created) == 18


def test_token_budget_counts_all_active_sets_and_preserves_priority(grid):
    plan, directory, state, client, _ = started(grid)
    client.finish()
    batch.collect(client, plan, directory, state)
    next_set = state['sets'][1]
    budget = max(j['estimated_input_tokens'] for j in state['jobs'] if j['name'] in next_set['jobs'])
    rolling.submit_ready(client, plan, directory, state, token_budget=budget)
    assert len(active_sets(state)) == 1
    totals = Counter()
    for j in state['jobs']:
        if j['status'] != 'prepared' and not j.get('collected'):
            totals[j['model']] += j['estimated_input_tokens']
    assert all(n <= budget for n in totals.values())
    assert not state['sets'][1]['admitted']


def test_lost_create_response_in_admitted_set_recovers_without_duplicate(grid):
    plan, directory, state, client, _ = started(grid)
    client.finish()
    batch.collect(client, plan, directory, state)
    client.fail_create = True
    with pytest.raises(TimeoutError):
        rolling.submit_ready(client, plan, directory, state)
    state = batch.load_state(plan, directory)
    client.fail_create = False
    batch.collect(client, plan, directory, state)  # Reconcile the accepted but unrecorded ID.
    rolling.submit_ready(client, plan, directory, state)
    assert len(active_sets(state)) == 4
    assert len(client.created) == 15
    assert len({r['query_id'] for r in ex.read_journal(directory / 'api/requests.jsonl')}) == 120


def test_failed_set_stops_new_spending_and_remains_incomplete(grid):
    plan, directory, state, client, _ = started(grid)
    client.finish(fail=True)
    batch.collect(client, plan, directory, state)
    with pytest.raises(RuntimeError, match='failures require review'):
        rolling.submit_ready(client, plan, directory, state)
    assert len(client.created) == 3
    report = rolling.progress(plan, directory, state)
    assert report['responses_complete'] == 0


def test_rolling_loop_finishes_all_sets_with_fake_polling_and_can_restart(grid):
    plan, directory, state, client, _ = started(grid)
    sleeps = []
    def tick(seconds):
        sleeps.append(seconds)
        assert len(active_sets(state)) <= 4
        client.finish()
    rolling.run_rolling(client, plan, directory, state, sleep=tick)
    assert sleeps
    assert len(ex.read_journal(directory / 'api/requests.jsonl')) == 480
    assert len(ex.read_journal(directory / 'api/responses.jsonl')) == 480
    assert all(j.get('collected') for j in state['jobs'])
    before = len(client.created)
    rolling.run_rolling(client, plan, directory, batch.load_state(plan, directory), sleep=lambda n: pytest.fail('Already done'))
    assert len(client.created) == before
    assert rolling.progress(plan, directory, state)['responses_complete'] == 16


def test_set_manifest_tampering_and_mismatched_samples_rejected(grid):
    plan, directory, state, client, _ = started(grid)
    state['sets'][0], state['sets'][1] = state['sets'][1], state['sets'][0]
    batch.save_state(directory, state)
    with pytest.raises(ValueError, match='order or membership'):
        batch.load_state(plan, directory)
    altered = deepcopy(plan)
    q = altered['queries'][15]
    altered['samples'][q['sample_id']]['ground_truth'] = [2, 0, 1]
    with pytest.raises(ValueError, match='not paired'):
        rolling.comparison_sets(altered)
