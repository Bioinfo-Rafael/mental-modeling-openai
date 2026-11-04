"""Export measured sequential call latency without treating it as model compute time."""
import statistics
from . import execution as ex, evaluation


def export(plan, directory):
    evaluation.evaluate_saved(plan, directory)
    responses = {r['query_id']: r for r in ex.read_journal(directory/'api/responses.jsonl')}
    attempted = {r['query_id'] for r in ex.read_journal(directory/'api/requests.jsonl')}
    rows = []
    for q in plan['queries']:
        r = responses.get(q['query_id'], {})
        raw = r.get('raw_response')
        usage = raw.get('usage', {}) if raw else {}
        parsed = evaluation.parse_intentions(raw, plan['prompt_spec']['aircraft_ids']) if raw else None
        rows.append(dict(query_id=q['query_id'], model=q['model'], repeat=q['repeat_id'],
                         center_step=plan['samples'][q['sample_id']]['target_step'],
                         status='received' if raw else ('error' if r else 'unknown' if q['query_id'] in attempted else 'pending'),
                         started_at=r.get('started_at'), finished_at=r.get('finished_at'),
                         elapsed_seconds=r.get('elapsed_seconds'), format_valid=parsed['valid'] if parsed else None,
                         request_id=r.get('request_id'), prompt_tokens=usage.get('prompt_tokens'),
                         cached_tokens=usage.get('prompt_tokens_details', {}).get('cached_tokens'),
                         completion_tokens=usage.get('completion_tokens'),
                         reasoning_tokens=usage.get('completion_tokens_details', {}).get('reasoning_tokens'),
                         error_type=r.get('error_type'), error=r.get('error')))
    seconds = [r['elapsed_seconds'] for r in rows if r['status']=='received' and r['elapsed_seconds'] is not None]
    summary = dict(transport='synchronous', concurrency=1, sdk_max_retries=0,
                   measurement='SDK call start to full response return; includes network, server wait, inference and SDK response parsing; excludes journal writes and model_dump',
                   planned=len(rows), attempted=len(attempted), responses_saved=len(seconds),
                   valid_format_responses=sum(r['format_valid'] is True for r in rows),
                   unattempted=len(rows)-len(attempted), unresolved=len(attempted)-len(seconds),
                   status='complete' if len(seconds)==len(rows) else 'incomplete',
                   mean_seconds=statistics.mean(seconds) if seconds else None,
                   median_seconds=statistics.median(seconds) if seconds else None,
                   min_seconds=min(seconds) if seconds else None, max_seconds=max(seconds) if seconds else None)
    ex.write_json(directory/'evaluation/timing.json', dict(summary=summary, requests=rows))
    return summary
