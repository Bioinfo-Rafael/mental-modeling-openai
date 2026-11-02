"""Two-aircraft tactical prose on frozen positive gates; latency is metadata only."""
import csv
import difflib
import json
from pathlib import Path
import shutil
import time

from . import workflow as w

SOURCE = w.ROOT / 'experiments/17_commentary_LAG/results/tactical_short_batch_v2'
MODEL = 'gpt-5.6-sol'


def adapt_system(original):
    replacements = {
        'The input contains 61 consecutive time points, from t-30 through t+30 inclusive, separated by 0.2 seconds.':
            'The input contains 40 consecutive time points ending at the current step t, separated by 0.2 seconds.',
        'The complete observation window spans 12 seconds.': 'The complete observation window spans 7.8 seconds.',
        'Use ALL 61 time points': 'Use ALL 40 time points',
        'The center step t is a reference point for constructing the window, NOT the exclusive subject of the interpretation.':
            'The current step t is the final observed step. Explain the supplied window; no observation after t is available.',
        'Treat the 61-step window as a continuous episode, not as 61 independent observations.':
            'Treat the 40-step window as a continuous episode, not as 40 independent observations.',
        'Output ONLY the most plausible intention narrative for each of the four aircraft.':
            'Output ONLY the most plausible intention narratives for A0100 and A0200. All four aircraft and missiles remain available as context.',
        'outside the four aircraft narratives': 'outside the two Team A aircraft narratives',
        'exactly four string fields': 'exactly two string fields',
        '  "A0200": "...",\n  "B0100": "...",\n  "B0200": "..."': '  "A0200": "..."',
        'use all 61 time points': 'use all 40 time points',
        'the four required JSON fields': 'the two required JSON fields',
    }
    for old, new in replacements.items():
        if original.count(old) != 1:
            raise ValueError(f'Unexpected source prompt: {old}')
        original = original.replace(old, new)
    return original


def create_plan(name, source_run):
    w.storage.safe_name(name)
    source_run = w.resolve_run(source_run)
    source_manifest = w.verify(source_run)
    records = w.records_for(source_run)
    if any(q['id'] not in records for q in source_manifest['queries'] if q['stage'] == 'gate'):
        raise ValueError('Source gate run is incomplete')
    selected = [q for q in source_manifest['queries'] if q['stage'] == 'gate'
                and records[q['id']]['api_status'] == 'ok'
                and records[q['id']]['choice'] in ('A1', 'A2', 'BOTH')]
    if not selected:
        raise ValueError('No positive gates')
    directory = w.RESULTS / name
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'prompts/source').mkdir(parents=True)
    for filename in ('system.txt', 'user.txt'):
        shutil.copy2(SOURCE / 'prompts/templates' / filename, directory / 'prompts/source' / filename)
    original = (directory / 'prompts/source/system.txt').read_text()
    system = adapt_system(original)
    (directory / 'prompts/system.txt').write_text(system)
    (directory / 'prompts/system.diff').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), system.splitlines(True), fromfile='17/system.txt', tofile='19/A1A2/system.txt')))
    original_user = (directory / 'prompts/source/user.txt').read_text()
    user = original_user.replace('Center step t: {center_step}', 'Current step t: {end_step}').replace(
        'Center time: {center_time_s:.1f}', 'Current time: {end_time_s:.1f}').replace(
        'Time points: 61', 'Time points: {time_points}').replace('Elapsed time: 12.0 seconds', 'Elapsed time: {elapsed_s:.1f} seconds').replace(
        'relative_step = step - t.', 'All observations end at t; no future observations are supplied.').replace(
        'explanation for each aircraft,', 'explanation for A0100 and A0200 only,')
    (directory / 'prompts/user.txt').write_text(user)
    (directory / 'prompts/user.diff').write_text(''.join(difflib.unified_diff(
        original_user.splitlines(True), user.splitlines(True), fromfile='17/user.txt', tofile='19/A1A2/user.txt')))
    (directory / 'inputs').mkdir()
    queries = []
    for gate in selected:
        filename = f"{gate['sample_id']}.json"
        shutil.copy2(source_run / 'inputs' / filename, directory / 'inputs' / filename)
        queries.append({**gate, 'id': gate['id'].replace('_gate', '_narrative'), 'stage': 'narrative',
                        'model': MODEL, 'request_sha256': None, 'gate_query_id': gate['id'],
                        'gate_choice': records[gate['id']]['choice'], 'gate_latency_s': records[gate['id']]['latency_s']})
    for q in queries:
        q['request_sha256'] = w.digest(request_for(directory, q))
    w.write_json(directory / 'source_gates.json', {q['id']: records[q['id']] for q in selected})
    shutil.copy2(Path(__file__), directory / 'implementation.py')
    manifest = {
        'schema_version': 1, 'kind': 'narrative', 'run_id': name, 'created_at': w.storage.utc_now(),
        'source_run': str(source_run), 'source_manifest_sha256': w.storage.file_hash(source_run / 'manifest.json'),
        'prompt_source': str(SOURCE.relative_to(w.ROOT)), 'model': MODEL, 'reasoning_effort': 'medium',
        'endpoint': '/v1/chat/completions', 'output_language': 'English (source prompt unchanged)',
        'timing_policy': 'measure_only_display_at_observation', 'transport_timeout_s': 180,
        'config': source_manifest['config'], 'recording_last_step': source_manifest['recording_last_step'],
        'repeats': source_manifest['repeats'], 'queries': queries,
        'files': {str(p.relative_to(directory)): w.storage.file_hash(p) for p in directory.rglob('*') if p.is_file()},
    }
    w.write_json(directory / 'manifest.json', manifest)
    (directory / 'manifest.sha256').write_text(w.storage.file_hash(directory / 'manifest.json') + '\n')
    export(directory)
    return directory


def request_for(directory, query):
    window = w.read_json(directory / 'inputs' / f"{query['sample_id']}.json")
    user = (directory / 'prompts/user.txt').read_text().format(**window, window_data=w.storage.canonical(window['blocks']))
    body = {'model': query['model'], 'reasoning_effort': 'medium', 'messages': [
        {'role': 'system', 'content': (directory / 'prompts/system.txt').read_text()},
        {'role': 'user', 'content': user}]}
    if query.get('request_sha256') and w.digest(body) != query['request_sha256']:
        raise ValueError(f"Request changed: {query['id']}")
    return body


def parse_output(raw):
    choice = raw['choices'][0]
    if choice.get('finish_reason') != 'stop' or choice['message'].get('refusal'):
        raise ValueError('Incomplete or refused output')
    result = json.loads(choice['message']['content'])
    if not isinstance(result, dict) or set(result) != set(w.UIDS):
        raise ValueError('Expected exactly A0100 and A0200')
    if any(not isinstance(v, str) or not v.strip() for v in result.values()):
        raise ValueError('Expected two nonempty strings')
    return result


def execute(directory, *, resume=False, retry_failed=False, limit=None, client=None):
    if retry_failed and not resume:
        raise ValueError('--retry-failed requires --resume')
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    with w.storage.run_lock(directory):
        manifest = w.verify(directory)
        if manifest.get('kind') != 'narrative':
            raise ValueError('Not a narrative run')
        records = w.records_for(directory)
        journal = w.storage.read_jsonl(directory / 'requests.jsonl')
        started = {r['query_id'] for r in journal}
        latest_attempt = {}
        for entry in journal:
            latest_attempt[entry['query_id']] = max(latest_attempt.get(entry['query_id'], 0), entry.get('attempt', 1))
        retryable = {'error', 'transport_timeout', 'invalid_output'}

        def needs_work(q):
            record = records.get(q['id'])
            return (not record or latest_attempt.get(q['id'], 0) > record.get('attempt', 1)
                    or (retry_failed and record['api_status'] in retryable))
        if started and not resume:
            raise ValueError('Run already started; use --resume. Started requests are never resent.')
        if not any(needs_work(q) for q in manifest['queries']):
            return
        owns_client = client is None
        if owns_client:
            from openai import OpenAI
            client = OpenAI(api_key=w.legacy_execution.read_api_key(), max_retries=0,
                            timeout=manifest['transport_timeout_s'])
        attempted = 0
        try:
            for q in manifest['queries']:
                if not needs_work(q):
                    continue
                old = records.get(q['id'])
                if q['id'] in started and (not old or latest_attempt[q['id']] > old.get('attempt', 1)):
                    w.write_json(directory / 'responses' / f"{q['id']}.json", {
                        'query_id': q['id'], 'api_status': 'uncertain', 'latency_s': None,
                        'attempt': latest_attempt[q['id']]})
                    continue
                if limit is not None and attempted >= limit:
                    break
                body = request_for(directory, q)
                if old:
                    w.write_json(directory / 'attempts' / q['id'] / f"attempt_{old.get('attempt', 1):03d}.json", old)
                attempt = max(latest_attempt.get(q['id'], 0), old.get('attempt', 1) if old else 0) + 1
                record = {'query_id': q['id'], 'attempt': attempt,
                          'started_at': w.storage.utc_now(), 'api_status': 'error'}
                w.storage.append_jsonl(directory / 'requests.jsonl', {**record, 'request_sha256': q['request_sha256'],
                    'implementation_sha256': w.storage.file_hash(Path(__file__))})
                start = time.perf_counter()
                try:
                    response = client.chat.completions.create(**body)
                    record['latency_s'] = time.perf_counter() - start
                    record['raw'] = response.model_dump(mode='json')
                    try:
                        record.update(api_status='ok', answers=parse_output(record['raw']))
                    except (ValueError, TypeError, KeyError, IndexError) as exc:
                        record.update(api_status='invalid_output', error=str(exc))
                except Exception as exc:
                    record['latency_s'] = time.perf_counter() - start
                    error = str(exc)
                    if getattr(client, 'api_key', None):
                        error = error.replace(client.api_key, '[REDACTED]')
                    record.update(api_status='transport_timeout' if type(exc).__name__ == 'APITimeoutError' else 'error',
                                  error=error, error_type=type(exc).__name__, http_status=getattr(exc, 'status_code', None))
                    cause, cause_types = exc.__cause__, []
                    while cause is not None and len(cause_types) < 5:
                        cause_types.append(type(cause).__name__)
                        cause = cause.__cause__
                    record['cause_types'] = cause_types
                record.update(finished_at=w.storage.utc_now(), over_3s=record['latency_s'] > 3)
                w.write_json(directory / 'responses' / f"{q['id']}.json", record)
                w.write_json(directory / 'attempts' / q['id'] / f'attempt_{attempt:03d}.json', record)
                w.storage.append_jsonl(directory / 'responses.jsonl', record)
                attempted += 1
                print(f"{q['id']}: {record['api_status']} / {record['latency_s']:.3f}s", flush=True)
                if record.get('error_type'):
                    print(f"  {record['error_type']} (causes: {', '.join(record.get('cause_types', [])) or 'unknown'})", flush=True)
                export(directory, manifest=manifest)
                if record['api_status'] == 'error':
                    break
        finally:
            if owns_client:
                client.close()
            export(directory, manifest=manifest)


def export(directory, *, manifest=None):
    manifest = manifest or w.verify(directory)
    if manifest.get('kind') != 'narrative':
        raise ValueError('Not a narrative run')
    records = w.records_for(directory)
    events, rows = [], []
    for q in manifest['queries']:
        r = records.get(q['id'], {})
        source_s = q['source_end_step'] * manifest['config']['dt_s']
        for uid in w.UIDS:
            row = {'query_id': q['id'], 'source_end_step': q['source_end_step'], 'source_time_s': source_s,
                   'aircraft_id': uid, 'gate_choice': q['gate_choice'], 'gate_latency_s': q['gate_latency_s'],
                   'api_status': r.get('api_status', 'pending'), 'latency_s': r.get('latency_s'),
                   'over_3s': r.get('over_3s'), 'answer': r.get('answers', {}).get(uid, ''),
                   'started_at': r.get('started_at'), 'finished_at': r.get('finished_at')}
            rows.append(row)
            if r.get('api_status') == 'ok':
                events.append({**row, 'id': q['id'] + '_' + uid, 'stage': 'narrative',
                               'repeat': q['repeat'], 'event_sim_s': source_s, 'label': row['answer']})
    events.sort(key=lambda e: (e['event_sim_s'], e['id']))
    ok = [r for r in records.values() if r['api_status'] == 'ok']
    times = [r['latency_s'] for r in ok]
    report = {'run_id': manifest['run_id'], 'planned_queries': len(manifest['queries']),
              'recorded_queries': len(records), 'ok_queries': len(ok), 'answer_count': len(events),
              'complete': len(records) == len(manifest['queries']),
              'all_successful': len(ok) == len(manifest['queries']),
              'p50_s': w.metrics.quantile(times, .5), 'p95_s': w.metrics.quantile(times, .95),
              'over_3s': sum(t > 3 for t in times), 'timing_policy': manifest['timing_policy']}
    w.write_json(directory / 'report/summary.json', report)
    with (directory / 'report/answers.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    replay = {'kind': 'narrative', 'run_id': manifest['run_id'], 'model': manifest['model'],
              'episode_id': manifest['config']['episode_id'], 'repeats': manifest['repeats'],
              'complete': report['complete'], 'summary': report, 'events': events,
              'last_step': manifest['recording_last_step'], 'timing_policy': manifest['timing_policy'],
              'timing_note': '応答時間は計測値の表示のみ。表示時刻は観測窓の末尾。遅延の加算・3秒での棄却はしません。'}
    w.write_json(directory / 'replay.json', replay)
    w.write_json(w.VIEWER / f"{manifest['run_id']}.json", replay)
    with w.storage.run_lock(w.VIEWER):
        index_path = w.VIEWER / 'index.json'
        index = w.read_json(index_path) if index_path.exists() else []
        index = [r for r in index if r['id'] != manifest['run_id']]
        index.append({'id': manifest['run_id'], 'kind': 'narrative', 'label': 'GPT-5.6 sol · A1/A2解説',
                      'episode_id': replay['episode_id'], 'created_at': manifest['created_at'],
                      'complete': report['complete'], 'url': f"/data/viz/viewer/commentary/{manifest['run_id']}.json"})
        w.write_json(index_path, sorted(index, key=lambda r: r['created_at'], reverse=True))
    return report
