"""Execute the reviewed few-shot requests with durable journals and no automatic retry."""
from __future__ import annotations

from importlib import import_module
import os
from pathlib import Path
import time

from experiments import common
from experiments.split_execution import execution_lock
from llm_x.data import Episode
from .scoring import score, CSV_FIELDS, VERSION


def journal(path):
    return common.read_jsonl(path) if path.exists() else []


def execute(manifest, *, resume=False):
    runner = import_module('experiments.08_fewshot.runner.fewshot_runner')
    root = common.output_path(runner.HERE / 'results')
    with execution_lock(root):
        return _execute(manifest, root, resume=resume)


def _execute(manifest, root, *, resume=False):
    if manifest['status'] != 'ready':
        raise ValueError('Input is not ready; run python run.py and inspect the preview')
    reviewed = common.read_json(root / 'dry_run/manifest.json')
    if reviewed['status'] != 'ready' or reviewed['queries'] != manifest['queries']:
        raise ValueError('Requests differ from the saved dry run; regenerate and review it first')
    if reviewed['input_sha256'] != manifest['input_sha256']:
        raise ValueError('Input files differ from the saved dry run')
    rows = manifest['queries']
    episodes = {}
    for row in rows:
        if row['request'] != common.expected_api_request(row['model'], row['system_prompt'], row['user_prompt']):
            raise ValueError('Request parameters differ from the planned prompts')
        path = common.ROOT / row['episode_path']
        if path not in episodes:
            episodes[path] = Episode.load(path)
        if episodes[path].sha256 != row['episode_sha256']:
            raise ValueError('Evaluation episode changed')
        score(row, '', episodes[path])  # Validate scoring before the first paid request.
    directory = root / 'api'
    if directory.exists():
        if not resume:
            raise FileExistsError('results/api already exists. Use --resume to send only unattempted queries')
        saved = common.read_json(directory / 'manifest.json')
        if saved['queries'] != rows or saved['input_sha256'] != manifest['input_sha256']:
            raise ValueError('Resume plan differs from the saved API run')
    elif resume:
        raise ValueError('No API run exists to resume')

    requests = journal(directory / 'requests.jsonl')
    responses = journal(directory / 'responses.jsonl')
    records = journal(directory / 'records.jsonl')
    if any(r.get('scoring_version') != VERSION for r in records):
        raise ValueError('Old scoring records found; run python analysis/rescore_results.py before resuming')
    attempted = {r['query_id'] for r in requests}
    completed = {r['query_id'] for r in records}
    lookup = {r['query_id']: r for r in rows}
    if len(attempted) != len(requests) or not attempted <= lookup.keys():
        raise ValueError('Invalid or duplicate request journal')
    if len(completed) != len(records) or not completed <= attempted:
        raise ValueError('Invalid result journal')
    pending = [row for row in rows if row['query_id'] not in attempted]
    if pending and not os.environ.get('OPENAI_API_KEY'):
        raise ValueError('OPENAI_API_KEY must be set in your terminal environment')
    if not directory.exists():
        directory.mkdir()
        common.write_json(directory / 'manifest.json', {**manifest, 'mode': 'execute',
                          'sdk_max_retries': 0, 'timeout_seconds': 180}, exclusive=True)

    def save_record(row, response):
        raw = response['raw_response']
        text = (raw.get('choices') or [{}])[0].get('message', {}).get('content')
        evaluation = score(row, text, episodes[common.ROOT / row['episode_path']])
        usage = raw.get('usage') or {}
        record = {**row, **response, **evaluation, 'assistant_text': text,
                  'input_tokens': usage.get('prompt_tokens'), 'output_tokens': usage.get('completion_tokens'),
                  'total_tokens': usage.get('total_tokens'), 'usage': raw.get('usage'),
                  'api_request_made': True, 'reused': False, 'source_experiment': '08_fewshot'}
        common.append_jsonl(directory / 'records.jsonl', record)
        records.append(record)
        completed.add(row['query_id'])

    # A response saved before an interruption can be scored locally without resending it.
    for response in responses:
        if response['query_id'] not in attempted:
            raise ValueError('Response without a journaled request')
        if response.get('raw_response') and response['query_id'] not in completed:
            save_record(lookup[response['query_id']], response)

    client = None
    try:
        if pending:
            from openai import OpenAI
            client = OpenAI(max_retries=0, timeout=180.0)
        for row in pending:
            qid = row['query_id']
            started = common.utc_now()
            common.append_jsonl(directory / 'requests.jsonl',
                                dict(query_id=qid, request_started_at_utc=started, kwargs=row['request']))
            attempted.add(qid)
            print(f"[{len(attempted)}/{len(rows)}] {row['condition_id']} query {row['ordinal'] + 1}", flush=True)
            clock = time.perf_counter()
            try:
                response = client.chat.completions.create(**row['request'])
                raw = response.model_dump(mode='json')
            except BaseException as exc:
                failure = dict(query_id=qid, request_started_at_utc=started,
                               request_elapsed_seconds=time.perf_counter()-clock,
                               raw_response=None, exception=common.exception_info(exc))
                common.append_jsonl(directory / 'responses.jsonl', failure)
                raise
            result = dict(query_id=qid, request_started_at_utc=started,
                          response_received_at_utc=common.utc_now(),
                          request_elapsed_seconds=time.perf_counter()-clock,
                          raw_response=raw, exception=None)
            common.append_jsonl(directory / 'responses.jsonl', result)
            save_record(row, result)
            print(f"  saved: {records[-1]['status']}", flush=True)
    finally:
        status = ('complete' if len(completed) == len(rows) else
                  'complete_with_errors' if len(attempted) == len(rows) else 'incomplete')
        summary = dict(status=status, planned_queries=len(rows), attempted_queries=len(attempted),
                       saved_responses=len(completed), unstarted_queries=len(rows)-len(attempted),
                       unresolved_queries=len(attempted-completed),
                       matches=sum(r['status']=='match' for r in records),
                       mismatches=sum(r['status']=='mismatch' for r in records),
                       ignored=sum(r['status']=='ignored' for r in records),
                       empty_responses=sum(r['status']=='empty_response' for r in records),
                       updated_at_utc=common.utc_now(), scoring_version=VERSION)
        common.write_json(directory / 'summary.json', summary)
        if records:
            common.write_csv(directory / 'records.csv', [{k:r.get(k) for k in CSV_FIELDS} for r in records])
        if client is not None:
            client.close()
    print(f"API run [{status}]: {len(completed)}/{len(rows)} responses saved. Results: {directory}")
    return 0 if status == 'complete' else 2
