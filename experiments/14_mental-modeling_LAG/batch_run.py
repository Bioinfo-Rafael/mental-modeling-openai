"""Resume a frozen LAG plan with Batch API after the synchronous prefix.

No prompt/plan regeneration. `prepare` is offline; `run` submits and collects.
Keep this adapter outside lag_experiment so the original source hashes stay valid.
"""
import argparse
from collections import Counter, defaultdict
import getpass
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(HERE), str(HERE.parent / '13_LAG'), str(ROOT / 'upstream/LLM-Xavier')]
from lag_experiment import execution as ex, evaluation

ENDPOINT = '/v1/chat/completions'
TERMINAL = {'completed', 'failed', 'expired', 'cancelled'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_plan(directory):
    plan = ex.read_json(directory / 'manifest.json')
    ex.verify_plan(plan)
    for relative, expected in plan['source_sha256'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f'Source changed since planning: {relative}')
    if ex.read_json(directory / 'api/identity.json')['plan_id'] != plan['plan_id']:
        raise ValueError('API identity differs from saved plan')
    return plan


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


def completion_query_ids(plan, after_sync):
    """Missing queries in conditions touched by the synchronous prefix only."""
    groups = defaultdict(list)
    for query in plan['queries']:
        sample = plan['samples'][query['sample_id']]
        key = tuple(sample[field] for field in ('episode_id', 'target_id', 'H', 'task'))
        key += (sample.get('observation_scope', 'self'),)
        groups[key].append(query['query_id'])
    prefix = {q['query_id'] for q in plan['queries'][:after_sync]}
    return {qid for ids in groups.values() if prefix.intersection(ids)
            for qid in ids if qid not in prefix}


def prepare(plan, directory, *, after_sync=220, max_requests=500, max_tokens=1_000_000,
            completion_only=False):
    target = directory / 'api/batch'
    if (target / 'state.json').exists():
        state = load_state(plan, directory)
        if completion_only != (state.get('scope', 'remaining') == 'prefix-completion'):
            raise ValueError('Existing Batch scope differs; do not overwrite or mix submission plans')
        return state
    if after_sync < 0 or not 1 <= max_requests <= 50_000 or max_tokens < 1:
        raise ValueError('Invalid Batch preparation limits')
    attempted, responded = journals(plan, directory)
    expected = [q['query_id'] for q in plan['queries'][:after_sync]]
    if list(attempted) != expected or set(attempted) != set(responded):
        raise ValueError(f'Need exactly the first {after_sync} synchronous requests with saved responses; '
                         f'found {len(attempted)} requests / {len(responded)} responses. Stop the original runner first.')
    if any(r.get('transport') == 'batch' for r in attempted.values()):
        raise ValueError('Batch requests exist without Batch state; recover state before proceeding')
    import tiktoken
    encoding = tiktoken.get_encoding('o200k_base')
    selected = completion_query_ids(plan, after_sync) if completion_only else None
    grouped = defaultdict(list)
    for query in plan['queries'][after_sync:]:
        if selected is not None and query['query_id'] not in selected:
            continue
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
    state = dict(version=1, plan_id=plan['plan_id'], after_sync=after_sync,
                 scope='prefix-completion' if completion_only else 'remaining',
                 prefix_requests_sha256=ex.digest(list(attempted.values())),
                 prefix_responses_sha256=ex.digest(list(responded.values())), jobs=jobs)
    save_state(directory, state)
    return state


def load_state(plan, directory):
    state = ex.read_json(directory / 'api/batch/state.json')
    if state['version'] != 1 or state['plan_id'] != plan['plan_id']:
        raise ValueError('Batch state identity mismatch')
    attempted, responded = journals(plan, directory)
    n = state['after_sync']
    prefix = [q['query_id'] for q in plan['queries'][:n]]
    if ex.digest([attempted[q] for q in prefix]) != state['prefix_requests_sha256']:
        raise ValueError('Synchronous request prefix changed')
    if ex.digest([responded[q] for q in prefix]) != state['prefix_responses_sha256']:
        raise ValueError('Synchronous response prefix changed')
    lookup = {q['query_id']: q for q in plan['queries']}
    seen = set(prefix)
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
    scope = state.get('scope', 'remaining')
    if scope not in ('remaining', 'prefix-completion', 'rolling-comparison'):
        raise ValueError('Unknown Batch scope')
    expected = (set(prefix) | completion_query_ids(plan, n)) if scope == 'prefix-completion' else set(lookup)
    if seen != expected:
        raise ValueError('Batch jobs do not partition the remaining plan')
    if scope == 'rolling-comparison':
        from batch_scheduler import validate_sets
        validate_sets(plan, state)
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
                           'No automatic resend. Check the dashboard; use adopt --job NAME --batch-id ID.')
    return remember_batch(directory, state, job, matches[0])


def submit_job(client, plan, directory, state, job, attempted=None):
    """Crash-safe submission shared by the legacy and rolling schedulers."""
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
    if state.get('scope') == 'rolling-comparison':
        raise ValueError('Use rolling for comparison-set scheduling')
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
    if state.get('scope') == 'rolling-comparison':
        from batch_scheduler import progress
        result['comparison_sets'] = progress(plan, directory, state)
    ex.write_json(directory / 'api/summary.json', result)
    evaluation.evaluate_saved(plan, directory)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'submit', 'complete-prefix', 'rolling', 'prepare-rolling', 'collect', 'status', 'adopt'))
    parser.add_argument('--output', default='sliding_window_v1')
    parser.add_argument('--after-sync', type=int, default=220)
    parser.add_argument('--max-requests', type=int, default=500)
    parser.add_argument('--max-input-tokens', type=int, default=1_000_000,
                        help='Conservative local token estimate per shard; not billing usage')
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--parallel-sets', type=int, default=4)
    parser.add_argument('--max-queued-input-tokens', type=int, default=1_000_000,
                        help='Local estimated input-token budget per model for rolling active sets')
    parser.add_argument('--job')
    parser.add_argument('--batch-id')
    parser.add_argument('--completion-only', action='store_true',
                        help='With prepare/run: restrict the plan to incomplete synchronous conditions')
    args = parser.parse_args(argv)
    if args.completion_only and args.command not in ('prepare', 'run'):
        parser.error('--completion-only is supported by prepare/run; complete-prefix already restricts its scope')
    if args.parallel_sets < 1 or args.max_queued_input_tokens < 1:
        parser.error('Parallel set count and queued token budget must be positive')
    if args.poll_seconds < 1:
        parser.error('--poll-seconds must be >= 1')
    if args.command == 'adopt' and (not args.job or not args.batch_id):
        parser.error('adopt requires --job and --batch-id')
    directory = (HERE / 'results' / args.output).resolve()
    if not directory.is_relative_to((HERE / 'results').resolve()) or directory == (HERE / 'results').resolve():
        parser.error('--output must identify a directory inside results/')
    with ex.run_lock(directory):
        plan = validate_plan(directory)
        if args.command in ('rolling', 'prepare-rolling'):
            from batch_scheduler import prepare_rolling
            state = prepare_rolling(plan, directory, after_sync=args.after_sync)
        else:
            state = prepare(plan, directory, after_sync=args.after_sync, max_requests=args.max_requests,
                        max_tokens=args.max_input_tokens,
                        completion_only=args.completion_only or args.command == 'complete-prefix'
                        ) if args.command in ('prepare', 'run', 'complete-prefix') else load_state(plan, directory)
        if args.command in ('prepare', 'status', 'prepare-rolling'):
            print(json.dumps(dict(synchronous_prefix=state['after_sync'],
                                  scope=state.get('scope', 'remaining'),
                                  remaining_requests=sum(len(j['query_ids']) for j in state['jobs']),
                                  jobs=[{k: j.get(k) for k in ('name', 'model', 'estimated_input_tokens', 'status', 'batch_id', 'collected')}
                                        | {'requests': len(j['query_ids'])} for j in state['jobs']]), indent=2))
            return 0
        from openai import OpenAI
        key = os.environ.get('OPENAI_API_KEY') or getpass.getpass('OpenAI API key: ').strip()
        if not key:
            raise ValueError('Empty API key; no requests sent')
        with OpenAI(api_key=key, max_retries=0, timeout=180) as client:
            try:
                if args.command == 'adopt':
                    job = next((j for j in state['jobs'] if j['name'] == args.job), None)
                    if job is None or job['status'] != 'submitting' or job.get('batch_id'):
                        raise ValueError('Only a submission with unknown Batch ID may be adopted')
                    remember_batch(directory, state, job, client.batches.retrieve(args.batch_id))
                elif args.command == 'rolling':
                    from batch_scheduler import run_rolling
                    run_rolling(client, plan, directory, state, parallel_sets=args.parallel_sets,
                                token_budget=args.max_queued_input_tokens, poll_seconds=args.poll_seconds)
                else:
                    while True:
                        collect(client, plan, directory, state)
                        if args.command in ('run', 'submit', 'complete-prefix'):
                            # Stop new spending after terminal failures; saved successes remain usable.
                            bad = [j['name'] for j in state['jobs'] if j.get('collected')
                                   and j['successful'] != len(j['query_ids'])]
                            if bad:
                                raise RuntimeError(f'Batch failures require review before more submissions: {bad}')
                            submit(client, plan, directory, state)
                        print(json.dumps(summarize(plan, directory, state)), flush=True)
                        if args.command != 'run' or all(j.get('collected') for j in state['jobs']):
                            break
                        time.sleep(args.poll_seconds)
            finally:
                summarize(plan, directory, state)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nStopped locally. Submitted batches continue remotely; rerun the same command to resume.', file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(ex._redact_error(exc, os.environ.get('OPENAI_API_KEY')), file=sys.stderr)
        raise SystemExit(1)
