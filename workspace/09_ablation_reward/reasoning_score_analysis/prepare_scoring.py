"""Extract public answer Reasoning for manual judging; never assign scores."""
import csv
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIELDS = ('query_id', 'model', 'Task', 'Metrics', 'H', 'shots', 'score_pattern',
          'ordinal', 'history_start', 'query_index', 'repeat_id', 'reward_present',
          'reward_scope', 'episode_path', 'Reasoning', 'score')


def extract(text):
    def heading(name):
        return re.compile(r'^\s*\d+\.\s*(?:\*\*)?\[?' + name
                          + r'\]?\s*:(?:\*\*)?[ \t]*', re.MULTILINE)
    starts = list(heading('Reasoning').finditer(text))
    ends = list(heading('Prediction').finditer(text))
    if len(starts) != 1 or len(ends) != 1 or starts[0].end() >= ends[0].start():
        raise ValueError('Missing, duplicate, or reversed Reasoning / Prediction heading')
    body = text[starts[0].end():ends[0].start()].strip()
    if not body:
        raise ValueError('Empty Reasoning')
    return body


def main():
    source = ROOT / 'results/api/records.jsonl'
    records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    manifest = json.loads((ROOT / 'results/api/manifest.json').read_text())
    ids = [r['query_id'] for r in records]
    if len(ids) != 96 or len(set(ids)) != 96 or set(ids) != {q['query_id'] for q in manifest['queries']}:
        raise ValueError('Expected all 96 unique planned responses')
    rows = []
    for record in records:
        row = {key: record[key] for key in FIELDS if key in record}
        row.update(model=record['model_alias'], Task=record['task'], Metrics=record['metric'],
                   reward_present=int(record['reward_present']),
                   Reasoning=extract(record['assistant_text']), score='')
        rows.append(row)
    destination = ROOT / 'reasoning_for_scoring.csv'
    # Never erase a manually edited input or a previously scored artifact.
    with destination.open('x', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    with destination.open(encoding='utf-8-sig', newline='') as handle:
        saved = list(csv.DictReader(handle))
    assert [r['query_id'] for r in saved] == ids
    assert all(a['Reasoning'] == b['Reasoning'] and b['score'] == '' for a, b in zip(rows, saved))
    report = dict(source='results/api/records.jsonl',
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  rows=len(saved), extracted_reasoning=len(saved), empty_scores=len(saved),
                  unique_query_ids=len(set(ids)), row_order_preserved=True,
                  csv_roundtrip_verified=True,
                  output_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    (HERE / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Prepared {len(saved)} unscored Reasoning rows: {destination}')


if __name__ == '__main__':
    main()
