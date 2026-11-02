"""Review exports, never treating hindsight narratives as ground truth."""
from collections import Counter
import csv
import re
from .common import ex, batch, answers


def export(plan, directory):
    ex.verify_plan(plan)
    batch.load_state(plan, directory)
    attempted, responded = batch.journals(plan, directory)
    rows = []
    for q in plan['queries']:
        sample = plan['samples'][q['sample_id']]
        record = responded.get(q['query_id'])
        raw = record.get('raw_response') if record else None
        parsed = answers.parse_intentions(raw) if raw is not None else None
        status = ('valid' if parsed['valid'] else 'format_error') if parsed else (
            'api_error' if record else ('pending' if q['query_id'] in attempted else 'unsubmitted'))
        narratives = parsed['narratives'] if parsed and parsed['valid'] else None
        sentences = {a: len([s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s])
                     for a, text in narratives.items()} if narratives else None
        rows.append(dict(**q, event=sample['event'], offset=sample['offset'],
                         condition_kind=sample['condition_kind'], target_interval=sample['target_interval'],
                         observation_range=[sample['start'], sample['observation_cutoff']],
                         other_events_in_target=sample['other_events_in_target'], status=status,
                         narratives=narratives, raw_response=raw,
                         error=(parsed['error'] if parsed else None) or (record or {}).get('error'),
                         usage=raw.get('usage') if raw else None,
                         length_metrics=answers.length_metrics(narratives, 55) if narratives else None,
                         sentence_counts_heuristic=sentences))
    summary = dict(plan_id=plan['plan_id'], counts=plan['counts'], statuses=dict(Counter(r['status'] for r in rows)),
                   excluded_events=plan['excluded_events'],
                   interpretation='n=30 is a hindsight reference, not ground truth or a forecast.',
                   latency='Batch latency is not online response latency.')
    out = directory / 'report'
    ex.write_json(out / 'responses.json', rows)
    ex.write_json(out / 'summary.json', summary)
    template = out / 'review_template.csv'
    if not template.exists():
        with template.open('w', newline='', encoding='utf-8-sig') as f:
            fields = ['query_id', 'event_id', 'target_aircraft', 'model', 'offset', 'condition_kind',
                      'behavior_match', 'purpose_support', 'unsupported_claims', 'reference_agreement',
                      'forecast_only', 'reviewer', 'notes']
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(query_id=row['query_id'], event_id=row['event']['id'],
                                     target_aircraft=row['event']['aircraft_id'], model=row['model'],
                                     offset=row['offset'], condition_kind=row['condition_kind']))
    return rows, summary
