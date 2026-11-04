"""Batch-only transport adapted from Experiment 14; no synchronous prefix."""
from collections import Counter, defaultdict
import getpass
import hashlib
import json
import os
import time

from . import execution as ex, evaluation

ENDPOINT = '/v1/chat/completions'
TERMINAL = {'completed', 'failed', 'expired', 'cancelled'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_sources(plan, directory, repository_root):
    ex.verify_plan(plan)
    if ex.read_json(directory / 'manifest.json') != plan:
        raise ValueError('Saved plan differs')
    for relative, expected in plan['source_sha256'].items():
        if sha(repository_root / relative) != expected:
            raise ValueError(f'Source changed since planning: {relative}')


def journals(plan, directory):
    requests = ex.read_journal(directory / 'api/requests.jsonl')
    responses = ex.read_journal(directory / 'api/responses.jsonl')
    lookup = {q['query_id']: q for q in plan['queries']}
    attempted, responded = {}, {}
    for row in requests:
        qid = row['query_id']
        if qid in attempted or qid not in lookup or row['request'] != ex.request_for(plan, lookup[qid]):
            raise ValueError('Invalid or duplicate request journal entry')
        attempted[qid] = row
    for row in responses:
        qid = row['query_id']
        if qid in responded or qid not in attempted:
            raise ValueError('Invalid or duplicate response journal entry')
        responded[qid] = row
    return attempted, responded


def save_state(directory, state):
    ex.write_json(directory / 'api/batch/state.json', state)


def prepare(plan, directory, *, max_requests=500, max_tokens=1_000_000):
    ex.verify_plan(plan)
    target = directory / 'api/batch'
    if (target / 'state.json').exists():
        return load_state(plan, directory)
    if not 1 <= max_requests <= 50_000 or max_tokens < 1:
        raise ValueError('Invalid Batch preparation limits')
    identity = directory / 'api/identity.json'
    if identity.exists() and ex.read_json(identity) != dict(plan_id=plan['plan_id'], transport='batch'):
        raise ValueError('API identity differs or belongs to a synchronous run; use a new output')
    attempted, responded = journals(plan, directory)
    if attempted or responded:
        raise ValueError('Requests exist without Batch state; recover state before proceeding')
    ex.write_json(identity, dict(plan_id=plan['plan_id'], transport='batch'))
    import tiktoken
    encoding = tiktoken.get_encoding('o200k_base')
    grouped = defaultdict(list)
    for query in plan['queries']:
        body = ex.request_for(plan, query)
        row = dict(custom_id=query['query_id'], method='POST', url=ENDPOINT, body=body)
        line = ex.canonical(row) + '\n'
        # Conservative planning estimate, NOT measured/billed usage.
        tokens = int(len(encoding.encode(ex.canonical(body), disallowed_special=())) * 1.1) + 128
        if tokens > max_tokens or len(line.encode()) > 190_000_000:
            raise ValueError('One request exceeds the configured shard budget')
        grouped[query['model_alias']].append((query, line, tokens))
    target.mkdir(parents=True, exist_ok=True)
    jobs = []
    for alias, items in grouped.items():
        chunks, chunk, count_tokens, count_bytes = [], [], 0, 0
        for item in items:
            size = len(item[1].encode())
            if chunk and (len(chunk) >= max_requests or count_tokens + item[2] > max_tokens
                          or count_bytes + size > 190_000_000):
                chunks.append(chunk)
                chunk, count_tokens, count_bytes = [], 0, 0
            chunk.append(item)
            count_tokens += item[2]
            count_bytes += size
        if chunk:
            chunks.append(chunk)
        for index, chunk in enumerate(chunks):
            name = f'{alias}_{index:03d}'
            path = target / f'{name}.input.jsonl'
            data = ''.join(item[1] for item in chunk)
            if path.exists() and path.read_text() != data:
                raise ValueError(f'Existing Batch input differs: {path}')
            if not path.exists():
                path.write_text(data, encoding='utf-8')
            jobs.append(dict(name=name, model=chunk[0][0]['model'], input_file=path.name,
                             sha256=sha(path), query_ids=[x[0]['query_id'] for x in chunk],
                             estimated_input_tokens=sum(x[2] for x in chunk), status='prepared',
                             job_key=ex.digest([plan['plan_id'], name, sha(path)])))
    state = dict(version=1, plan_id=plan['plan_id'], transport='batch', jobs=jobs)
    save_state(directory, state)
    return state


def load_state(plan, directory):
    ex.verify_plan(plan)
    state = ex.read_json(directory / 'api/batch/state.json')
    if state['version'] != 1 or state['plan_id'] != plan['plan_id']:
        raise ValueError('Batch state identity mismatch')
    if ex.read_json(directory / 'api/identity.json') != dict(plan_id=plan['plan_id'], transport='batch'):
        raise ValueError('API identity differs')
    attempted, responded = journals(plan, directory)
    lookup = {q['query_id']: q for q in plan['queries']}
    seen = set()
    for job in state['jobs']:
        path = directory / 'api/batch' / job['input_file']
        if path.parent != directory / 'api/batch' or sha(path) != job['sha256']:
            raise ValueError('Batch input file hash/path mismatch')
        rows = ex.read_journal(path)
        if [r['custom_id'] for r in rows] != job['query_ids']:
            raise ValueError('Batch query IDs differ from input file')
        if job['job_key'] != ex.digest([plan['plan_id'], job['name'], sha(path)]):
            raise ValueError('Batch job identity mismatch')
        for row in rows:
            qid = row['custom_id']
            if qid in seen or qid not in lookup or row != dict(
                    custom_id=qid, method='POST', url=ENDPOINT, body=ex.request_for(plan, lookup[qid])):
                raise ValueError('Batch request differs from frozen plan')
            if row['body']['model'] != job['model']:
                raise ValueError('Mixed models in Batch input')
            seen.add(qid)
            if qid in attempted and attempted[qid].get('batch_job_key') != job['job_key']:
                raise ValueError('A Batch candidate was sent by another runner; refusing duplicate submission')
    if seen != set(lookup):
        raise ValueError('Batch jobs do not partition the whole plan')
    return state


def remember_batch(directory, state, job, batch):
    data = batch.model_dump(mode='json')
    if data['input_file_id'] != job['file_id'] or data['endpoint'] != ENDPOINT:
        raise ValueError('Remote Batch input/endpoint mismatch')
    if (data.get('metadata') or {}).get('lag_job_key') != job['job_key']:
        raise ValueError('Remote Batch metadata mismatch')
    if job.get('batch_id') and job['batch_id'] != data['id']:
        raise ValueError('Remote Batch ID mismatch')
    job.update(batch_id=data['id'], status=data['status'])
    ex.write_json(directory / 'api/batch' / f"{job['name']}.remote.json", data)
    save_state(directory, state)
    return data


def recover_submission(client, directory, state, job):
    matches = []
    # Bounded scan, with an explicit adopt command for older jobs.
    for index, batch in enumerate(client.batches.list(limit=100)):
        if (batch.metadata or {}).get('lag_job_key') == job['job_key']:
            matches.append(batch)
        if index >= 999:
            break
    if len(matches) != 1:
        raise RuntimeError(f"Ambiguous submission for {job['name']}: {len(matches)} matching batches. "
                           'No automatic resend. Check the dashboard; use --adopt --job NAME --batch-id ID.')
    return remember_batch(directory, state, job, matches[0])


def submit_job(client, plan, directory, state, job, attempted=None):
    """Persist submission intent before calling the remote Batch API."""
    if job['status'] != 'prepared':
        return
    if attempted is None:
        attempted, _ = journals(plan, directory)
    if not job.get('file_id'):
        with (directory / 'api/batch' / job['input_file']).open('rb') as handle:
            job['file_id'] = client.files.create(file=handle, purpose='batch').id
        save_state(directory, state)
    lookup = {q['query_id']: q for q in plan['queries']}
    # Persist reservations and submission intent before calling the paid API.
    for qid in job['query_ids']:
        if qid in attempted:
            if attempted[qid].get('batch_job_key') != job['job_key']:
                raise ValueError('Request already attempted outside this Batch')
            continue
        row = dict(query_id=qid, request=ex.request_for(plan, lookup[qid]),
                   started_at=ex._now(), transport='batch', batch_job_key=job['job_key'])
        ex.append_journal(directory / 'api/requests.jsonl', row)
        attempted[qid] = row
    job['status'] = 'submitting'
    save_state(directory, state)
    remote = client.batches.create(input_file_id=job['file_id'], endpoint=ENDPOINT,
                                   completion_window='24h', metadata={'lag_job_key': job['job_key']})
    remember_batch(directory, state, job, remote)
    print(f"Submitted {job['name']}: {len(job['query_ids'])} requests, {job['batch_id']}", flush=True)


def submit(client, plan, directory, state):
    attempted, _ = journals(plan, directory)
    busy = {j['model'] for j in state['jobs'] if j['status'] != 'prepared' and not j.get('collected')}
    for job in state['jobs']:
        if job['status'] != 'prepared' or job['model'] in busy:
            continue
        submit_job(client, plan, directory, state, job, attempted)
        busy.add(job['model'])


def collect_job(client, directory, state, job, remote):
    if remote['status'] not in TERMINAL:
        return False
    expected = set(job['query_ids'])
    rows = {}
    for kind in ('output', 'error'):
        file_id = remote.get(f'{kind}_file_id')
        if not file_id:
            continue
        path = directory / 'api/batch' / f"{job['name']}.{kind}.jsonl"
        if not path.exists():
            data = client.files.content(file_id).text
            pending = path.with_suffix('.tmp')
            pending.write_text(data, encoding='utf-8')
            pending.replace(path)
        for row in ex.read_journal(path):
            qid = row['custom_id']
            if qid not in expected or qid in rows:
                raise ValueError('Unknown or duplicate custom_id in Batch output')
            rows[qid] = row
    # Validate the entire payload before appending anything. Missing requests are
    # retained as explicit terminal errors, never silently resent.
    saved = {r['query_id']: r for r in ex.read_journal(directory / 'api/responses.jsonl')}
    pending_records = []
    for qid in job['query_ids']:
        row = rows.get(qid, {})
        response = row.get('response') or {}
        error = row.get('error')
        raw = response.get('body') if response.get('status_code') == 200 and not error else None
        if raw is not None and not isinstance(raw, dict):
            raise ValueError('Invalid Batch response body')
        if qid in saved:
            if saved[qid].get('batch_id') != job['batch_id'] or saved[qid]['raw_response'] != raw:
                raise ValueError('Existing response conflicts with Batch result')
            continue
        record = dict(query_id=qid, raw_response=raw, transport='batch', batch_id=job['batch_id'],
                      finished_at=ex._now(), elapsed_seconds=None,
                      error_type=None if raw is not None else 'BatchRequestError',
                      error=None if raw is not None else ex.canonical(error or response or
                          {'code': f"batch_{remote['status']}_missing_result", 'details': remote.get('errors')}))
        pending_records.append(record)
    for record in pending_records:
        ex.append_journal(directory / 'api/responses.jsonl', record)
    job['collected'] = True
    job['successful'] = sum(1 for r in rows.values() if (r.get('response') or {}).get('status_code') == 200
                            and not r.get('error') and isinstance(r['response'].get('body'), dict))
    save_state(directory, state)
    print(f"Collected {job['name']}: {job['successful']}/{len(expected)} successful ({remote['status']})", flush=True)
    return True


def collect(client, plan, directory, state):
    changed = False
    for job in state['jobs']:
        if job.get('collected') or job['status'] == 'prepared':
            continue
        if not job.get('batch_id'):
            remote = recover_submission(client, directory, state, job)
        else:
            remote = remember_batch(directory, state, job, client.batches.retrieve(job['batch_id']))
        print(f"{job['name']}: {job['status']}", flush=True)
        changed = collect_job(client, directory, state, job, remote) or changed
    return changed


def summarize(plan, directory, state):
    attempted, responded = journals(plan, directory)
    success = sum(r['raw_response'] is not None for r in responded.values())
    result = dict(planned=len(plan['queries']), attempted=len(attempted), responses_saved=success,
                  unattempted=len(plan['queries'])-len(attempted), unresolved=len(attempted)-success,
                  status='complete' if success == len(plan['queries']) else 'incomplete',
                  batch_jobs=dict(Counter(j['status'] for j in state['jobs'])),
                  batch_collected=sum(bool(j.get('collected')) for j in state['jobs']))
    ex.write_json(directory / 'api/summary.json', result)
    evaluation.evaluate_saved(plan, directory)
    return result


def local_status(state):
    return dict(transport='batch', requests=sum(len(j['query_ids']) for j in state['jobs']),
                jobs=[{key: job.get(key) for key in ('name', 'model', 'status', 'batch_id', 'collected')}
                      | {'requests': len(job['query_ids'])} for job in state['jobs']])


def execute(plan, directory, *, repository_root, submit_new=True, wait=False,
            poll_seconds=30, client_factory=None, adopt_job=None, batch_id=None):
    """Submit once or collect; repeated invocations never recreate known batches."""
    if not 1 <= poll_seconds <= 60:
        raise ValueError('poll_seconds must be between 1 and 60')
    ex.verify_plan(plan)
    if submit_new:
        validate_sources(plan, directory, repository_root)
        state = prepare(plan, directory)
    else:
        state = load_state(plan, directory)
    if all(job.get('collected') for job in state['jobs']) and not adopt_job:
        return summarize(plan, directory, state)
    if not submit_new and not adopt_job and all(job['status'] == 'prepared' for job in state['jobs']):
        raise ValueError('No Batch has been submitted; use --execute first')
    key, client = None, None
    try:
        if client_factory is not None:
            client = client_factory()
        else:
            from openai import OpenAI
            key = os.environ.get('OPENAI_API_KEY') or getpass.getpass('OpenAI API key: ').strip()
            if not key:
                raise ValueError('Empty API key; no requests sent')
            client = OpenAI(api_key=key, max_retries=0, timeout=180)
        if adopt_job:
            job = next((j for j in state['jobs'] if j['name'] == adopt_job), None)
            if job is None or job['status'] != 'submitting' or job.get('batch_id'):
                raise ValueError('Only a submission with unknown Batch ID may be adopted')
            remember_batch(directory, state, job, client.batches.retrieve(batch_id))
            return summarize(plan, directory, state)
        while True:
            collect(client, plan, directory, state)
            if submit_new:
                bad = [j['name'] for j in state['jobs'] if j.get('collected')
                       and j['successful'] != len(j['query_ids'])]
                if bad:
                    raise RuntimeError(f'Batch failures require review before more submissions: {bad}')
                submit(client, plan, directory, state)
            result = summarize(plan, directory, state)
            print(json.dumps(result), flush=True)
            active = any(j['status'] != 'prepared' and not j.get('collected') for j in state['jobs'])
            if not wait or all(j.get('collected') for j in state['jobs']) or not active:
                return result
            time.sleep(poll_seconds)
    except Exception as exc:
        # A key entered via getpass must not leak through an SDK error traceback.
        if key:
            raise RuntimeError(ex._redact_error(exc, key)) from None
        raise
    finally:
        if client is not None:
            client.close()
        summarize(plan, directory, state)
