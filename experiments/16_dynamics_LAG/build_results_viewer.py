"""Build a standalone offline LAG viewer from an append-safe journal snapshot.

Read-only with respect to experiment inputs/results. No lock or API is needed;
only the destination HTML is written. Shared prompts are embedded once/sample.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(HERE), str(HERE.parent / '13_LAG'), str(ROOT / 'upstream/LLM-Xavier')]
from dynamics_experiment import execution as ex, evaluation


def journal_snapshot(path):
    content = path.read_bytes() if path.exists() else b''
    # An active append may not have written the final newline yet. Include only
    # fully committed lines; never discard malformed *complete* lines.
    end = content.rfind(b'\n') + 1
    complete = content[:end]
    rows = [json.loads(line) for line in complete.splitlines() if line.strip()]
    return rows, dict(sha256=hashlib.sha256(complete).hexdigest(), bytes=len(complete),
                     rows=len(rows), trailing_bytes_excluded=len(content)-len(complete))


def index(rows):
    result = {}
    for row in rows:
        if row['query_id'] in result:
            raise ValueError('Duplicate query_id in journal')
        result[row['query_id']] = row
    return result


def load_data(directory):
    source = directory / 'manifest.json'
    manifest_bytes = source.read_bytes()
    plan = json.loads(manifest_bytes)
    ex.verify_plan(plan)
    identity = directory / 'api/identity.json'
    if identity.exists() and ex.read_json(identity)['plan_id'] != plan['plan_id']:
        raise ValueError('API identity differs from the saved plan')
    # Read responses BEFORE requests: every response must already have a request.
    response_rows, response_info = journal_snapshot(directory / 'api/responses.jsonl')
    request_rows, request_info = journal_snapshot(directory / 'api/requests.jsonl')
    if (response_rows or request_rows) and not identity.exists():
        raise ValueError('API identity is missing')
    responses, requests = index(response_rows), index(request_rows)
    lookup = {q['query_id']: q for q in plan['queries']}
    if not set(responses) <= set(requests) <= set(lookup):
        raise ValueError('Unknown query or response without a request')
    for qid, request in requests.items():
        if request['request'] != ex.request_for(plan, lookup[qid]):
            raise ValueError('Saved request differs from the frozen plan')
    rows = []
    for query in plan['queries']:
        qid = query['query_id']
        sample = plan['samples'][query['sample_id']]
        response, request = responses.get(qid), requests.get(qid)
        raw = response.get('raw_response') if response else None
        text = evaluation.response_text(raw)
        scoring = None
        if response is None:
            status = 'pending' if request else 'unstarted'
        elif raw is None:
            status = 'api_error'
        else:
            scoring = evaluation.score(sample['prediction_fields'], sample['ground_truth'], text)
            if not scoring['parse_success']:
                status = 'parse_error'
            else:
                status = 'trajectory_scored'
        row = dict(query, **{f: sample[f] for f in ('episode_id', 'target_id', 'H', 'task', 'observation_scope',
                                                    'sample_number', 'start', 'target_step', 'target_steps', 'horizon', 'problem_id')},
                   status=status, raw_response=raw, scoring=scoring,
                   transport=(response or request or {}).get('transport', 'sync' if request else None),
                   batch_id=(response or {}).get('batch_id'),
                   started_at=(request or {}).get('started_at'),
                   finished_at=(response or {}).get('finished_at'),
                   elapsed_seconds=(response or {}).get('elapsed_seconds'),
                   error_type=(response or {}).get('error_type'), error=(response or {}).get('error'))
        # Pair only the same episode, target, history, task and actual time window.
        row['comparison_id'] = ex.digest([sample[f] for f in ('episode_id', 'target_id', 'H', 'task',
                                                              'start', 'target_step', 'sample_number', 'problem_id')])
        rows.append(row)
    counts = Counter(r['status'] for r in rows)
    return dict(created_at_jst=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds'),
                run=directory.name, plan_id=plan['plan_id'], rows=rows, samples=plan['samples'],
                request_options=plan['request_options'], scoring_version=evaluation.SCORING_VERSION,
                counts=dict(counts), sources=dict(manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                                                  responses=response_info, requests=request_info))


def render_html(data, template):
    if template.count('__DATASET__') != 1:
        raise ValueError('Expected exactly one dataset placeholder')
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return template.replace('__DATASET__', payload.replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='dynamics_batch_v1')
    parser.add_argument('--output', type=Path, default=HERE / 'view_rawdata.html')
    parser.add_argument('--overwrite', action='store_true', help='Replace only an existing generated HTML')
    args = parser.parse_args(argv)
    directory = (HERE / 'results' / args.run).resolve()
    if not directory.is_relative_to((HERE / 'results').resolve()):
        parser.error('--run must be inside results/')
    output = args.output.resolve()
    if output.suffix.lower() != '.html' or output.is_relative_to((HERE / 'results').resolve()):
        parser.error('Output must be an HTML outside the source results directory')
    if output.exists() and not args.overwrite:
        parser.error('HTML already exists; use --overwrite to refresh this snapshot')
    data = load_data(directory)
    timeline = directory / 'analysis/trajectory_timelines_v1/index.html'
    data['timeline_url'] = os.path.relpath(timeline, output.parent) if timeline.exists() else None
    html = render_html(data, (HERE / 'templates/results_viewer_template.html').read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        pending = output.with_suffix('.html.tmp')
        pending.write_text(html, encoding='utf-8')
        pending.replace(output)
    else:
        with output.open('x', encoding='utf-8') as handle:
            handle.write(html)
    print(json.dumps(dict(output=str(output), planned=len(data['rows']), statuses=data['counts'],
                          size_mb=round(output.stat().st_size/1e6, 2)), ensure_ascii=False))


if __name__ == '__main__':
    main()
