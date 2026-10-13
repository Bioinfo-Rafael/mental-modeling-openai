"""Rolling 30-query comparison sets over the immutable experiment plan.

Order: episode -> target -> history -> task; both observation scopes and all
three models stay in one logical set. No data generation or API calls at import.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import json
import time

import batch_run as batch
from lag_experiment import execution as ex, evaluation


def comparison_sets(plan):
    groups = {}
    for q in plan['queries']:
        s = plan['samples'][q['sample_id']]
        key = tuple(s[f] for f in ('episode_id', 'target_id', 'H', 'task'))
        groups.setdefault(key, []).append(q)
    settings = plan['experiment']
    ranks = [dict(zip(settings[field], range(len(settings[field]))))
             for field in ('episodes', 'target_ids', 'histories', 'tasks')]
    result = []
    for key, queries in sorted(groups.items(), key=lambda item: tuple(r[v] for r, v in zip(ranks, item[0]))):
        slots = defaultdict(dict)
        for q in queries:
            s = plan['samples'][q['sample_id']]
            slot = (s['observation_scope'], s['sample_number'])
            if q['model_alias'] in slots[slot] or q['repeat_id'] != 1:
                raise ValueError('Repeated query within a comparison set')
            slots[slot][q['model_alias']] = q['sample_id']
        expected = {(scope, n) for scope in ('self', 'all_aircraft_missiles') for n in range(5)}
        if set(slots) != expected or any(set(v) != {'sol', 'terra', 'luna'} or len(set(v.values())) != 1
                                        for v in slots.values()):
            raise ValueError('Rolling scheduling requires matching 3 models x 2 scopes x n=5')
        for n in range(5):
            left = plan['samples'][slots['self', n]['sol']]
            right = plan['samples'][slots['all_aircraft_missiles', n]['sol']]
            if any(left[field] != right[field] for field in ('problem_id', 'start', 'target_step', 'ground_truth')):
                raise ValueError('self/all-aircraft questions are not paired')
        result.append(dict(set_id=ex.digest(key)[:24],
                           condition=dict(zip(('episode_id', 'target_id', 'H', 'task'), key)),
                           query_ids=[q['query_id'] for q in queries]))
    return result


def validate_sets(plan, state):
    expected = comparison_sets(plan)
    if len(expected) != len(state['sets']):
        raise ValueError('Comparison set count changed')
    lookup = {q: s['set_id'] for s in expected for q in s['query_ids']}
    names = set()
    for job in state['jobs']:
        if job['name'] in names or {lookup[q] for q in job['query_ids']} != {job.get('set_id')}:
            raise ValueError('Batch crosses comparison sets or has duplicate name')
        names.add(job['name'])
    for stored, wanted in zip(state['sets'], expected):
        if any(stored[k] != wanted[k] for k in wanted):
            raise ValueError('Comparison set order or membership changed')
        associated = {j['name'] for j in state['jobs'] if j['set_id'] == stored['set_id']}
        if set(stored['jobs']) != associated or len(stored['jobs']) != len(associated):
            raise ValueError('Comparison set jobs changed')
        if any(j['status'] != 'prepared' for j in state['jobs'] if j['name'] in associated) and not stored['admitted']:
            raise ValueError('Submitted set lost its admission record')
    if not set(state['bootstrap_jobs']) <= names:
        raise ValueError('Unknown bootstrap jobs')


def prepare_rolling(plan, directory, *, after_sync=220):
    groups = comparison_sets(plan)
    target = directory / 'api/batch'
    if (target / 'state.json').exists():
        state = batch.load_state(plan, directory)
    else:
        state = batch.prepare(plan, directory, after_sync=after_sync, completion_only=True)
    if state.get('scope') == 'rolling-comparison':
        return state
    if state.get('scope') != 'prefix-completion':
        raise ValueError('Rolling migration requires the prefix-completion plan; retain any other live batches')
    # Preserve the exact live Batch IDs, input hashes, and journal associations.
    backup = target / 'state.before_rolling.json'
    if not backup.exists():
        ex.write_json(backup, state)
    state = deepcopy(state)
    state.update(scope='rolling-comparison', scheduler_version=1, sets=[],
                 bootstrap_jobs=[j['name'] for j in state['jobs']])
    prefix = {q['query_id'] for q in plan['queries'][:state['after_sync']]}
    covered = prefix | {qid for j in state['jobs'] for qid in j['query_ids']}
    lookup = {q['query_id']: q for q in plan['queries']}
    set_for_query = {qid: s['set_id'] for s in groups for qid in s['query_ids']}
    for job in state['jobs']:
        ids = {set_for_query[qid] for qid in job['query_ids']}
        if len(ids) != 1:
            raise ValueError('An existing batch spans multiple comparison sets')
        job['set_id'] = ids.pop()
    import tiktoken
    encoding = tiktoken.get_encoding('o200k_base')
    for index, group in enumerate(groups):
        by_model = defaultdict(list)
        for qid in group['query_ids']:
            if qid not in covered:
                by_model[lookup[qid]['model_alias']].append(lookup[qid])
        for alias, queries in by_model.items():
            name = f'set{index:03d}_{alias}'
            rows, tokens = [], 0
            for query in queries:
                body = ex.request_for(plan, query)
                rows.append(dict(custom_id=query['query_id'], method='POST', url=batch.ENDPOINT, body=body))
                tokens += int(len(encoding.encode(ex.canonical(body), disallowed_special=())) * 1.1) + 128
            data = ''.join(ex.canonical(row) + '\n' for row in rows)
            if len(data.encode()) > 190_000_000:
                raise ValueError('A comparison batch exceeds the file-size budget')
            path = target / f'{name}.input.jsonl'
            if path.exists() and path.read_text() != data:
                raise ValueError(f'Existing comparison input differs: {path}')
            if not path.exists():
                pending = path.with_suffix('.tmp')
                pending.write_text(data, encoding='utf-8')
                pending.replace(path)
            state['jobs'].append(dict(name=name, model=queries[0]['model'], input_file=path.name,
                                      sha256=batch.sha(path), query_ids=[q['query_id'] for q in queries],
                                      estimated_input_tokens=tokens, status='prepared', set_id=group['set_id'],
                                      job_key=ex.digest([plan['plan_id'], name, batch.sha(path)])))
        jobs = [j for j in state['jobs'] if j['set_id'] == group['set_id']]
        state['sets'].append(dict(**group, jobs=[j['name'] for j in jobs],
                                  admitted=any(j['status'] != 'prepared' for j in jobs)))
    validate_sets(plan, state)
    batch.save_state(directory, state)
    return batch.load_state(plan, directory)


def done(group, jobs):
    return all(jobs[name].get('collected') for name in group['jobs'])


def fail_if_errors(state):
    bad = [j['name'] for j in state['jobs'] if j.get('collected') and j['successful'] != len(j['query_ids'])]
    if bad:
        raise RuntimeError(f'Batch failures require review before more submissions: {bad}; use collect for other live batches')


def submit_ready(client, plan, directory, state, *, parallel_sets=4, token_budget=1_000_000):
    if parallel_sets < 1 or token_budget < 1:
        raise ValueError('Positive parallel-set count and token budget required')
    fail_if_errors(state)
    jobs = {j['name']: j for j in state['jobs']}
    attempted, _ = batch.journals(plan, directory)
    submitted = 0
    # Honor the previously requested five-query completion before opening new sets.
    boot = [jobs[n] for n in state['bootstrap_jobs']]
    if any(not j.get('collected') for j in boot):
        pending_tokens = Counter()
        for j in boot:
            if not j.get('collected'):
                pending_tokens[j['model']] += j['estimated_input_tokens']
        if any(n > token_budget for n in pending_tokens.values()):
            raise ValueError('Bootstrap requires a larger per-model token budget')
        for job in boot:
            if job['status'] == 'prepared':
                group = next(s for s in state['sets'] if s['set_id'] == job['set_id'])
                group['admitted'] = True
                batch.save_state(directory, state)
                batch.submit_job(client, plan, directory, state, job, attempted)
                submitted += 1
        return submitted
    active = [s for s in state['sets'] if s['admitted'] and not done(s, jobs)]
    reserved = Counter()
    for s in active:
        for name in s['jobs']:
            job = jobs[name]
            if not job.get('collected'):
                reserved[job['model']] += job['estimated_input_tokens']
    # Finish partially submitted sets first after an interrupted process.
    for group in active:
        for name in group['jobs']:
            if jobs[name]['status'] == 'prepared':
                if any(n > token_budget for n in reserved.values()):
                    raise ValueError('Active reservations exceed the requested token budget; keep prior budget or collect first')
                batch.submit_job(client, plan, directory, state, jobs[name], attempted)
                submitted += 1
    for group in state['sets']:
        if group['admitted'] or done(group, jobs):
            continue
        if len(active) >= parallel_sets:
            break
        needed = Counter()
        for name in group['jobs']:
            job = jobs[name]
            if not job.get('collected'):
                needed[job['model']] += job['estimated_input_tokens']
        if any(n > token_budget for n in needed.values()):
            raise ValueError(f"Set {group['condition']} alone exceeds the per-model token budget")
        if any(reserved[m] + n > token_budget for m, n in needed.items()):
            break  # Preserve priority; do not skip ahead to cheaper/later sets.
        group['admitted'] = True
        batch.save_state(directory, state)
        active.append(group)
        reserved.update(needed)
        for name in group['jobs']:
            job = jobs[name]
            if job['status'] == 'prepared':
                batch.submit_job(client, plan, directory, state, job, attempted)
                submitted += 1
    return submitted


def progress(plan, directory, state):
    _, responded = batch.journals(plan, directory)
    jobs = {j['name']: j for j in state['jobs']}
    records = []
    for group in state['sets']:
        rows = [responded[q] for q in group['query_ids'] if q in responded]
        successes = sum(r['raw_response'] is not None for r in rows)
        parsed = sum(evaluation.parse_prediction(evaluation.response_text(r['raw_response']),
                                                group['condition']['task'])[0] is not None for r in rows)
        records.append(dict(set_id=group['set_id'], **group['condition'], planned=30, responses=len(rows),
                            successful=successes, parsed=parsed, responses_complete=successes == 30,
                            predictions_complete=parsed == 30,
                            active=group['admitted'] and not done(group, jobs)))
    ex.write_json(directory / 'api/batch/comparison_sets.json', records)
    return dict(total=len(records), responses_complete=sum(r['responses_complete'] for r in records),
                predictions_complete=sum(r['predictions_complete'] for r in records),
                active=sum(r['active'] for r in records))


def run_rolling(client, plan, directory, state, *, parallel_sets=4, token_budget=1_000_000,
                poll_seconds=30, sleep=time.sleep):
    first = True
    while True:
        changed = batch.collect(client, plan, directory, state)
        submitted = submit_ready(client, plan, directory, state, parallel_sets=parallel_sets, token_budget=token_budget)
        if first or changed or submitted:
            print(json.dumps(batch.summarize(plan, directory, state)), flush=True)
            first = False
        if all(j.get('collected') for j in state['jobs']):
            return
        sleep(poll_seconds)
