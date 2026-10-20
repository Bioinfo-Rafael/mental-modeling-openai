"""Build a self-contained offline viewer from a snapshot of the API run."""
import csv
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parents[1]


def read_journal(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_bytes().splitlines(keepends=True) if line.endswith(b'\n')]


def main():
    source=HERE/'results/api'
    manifest=json.loads((source/'manifest.json').read_text())
    records=read_journal(source/'records.jsonl')
    lookup={r['query_id']:r for r in records}
    responses={r['query_id']:r for r in read_journal(source/'responses.jsonl')}
    requested={r['query_id'] for r in read_journal(source/'requests.jsonl')}
    assert len(lookup)==len(records), 'Duplicate records'
    assert set(lookup)<={q['query_id'] for q in manifest['queries']}, 'Unexpected records'
    score_path=HERE/'reasoning_for_scoring_scored_astra_high.csv'
    scores={}
    if score_path.exists():
        with score_path.open(encoding='utf-8-sig', newline='') as handle:
            for score in csv.DictReader(handle):
                qid=score['query_id']
                if qid in scores:
                    raise ValueError(f'Duplicate reasoning score: {qid}')
                if score['score'] not in ('1','2','3','4'):
                    raise ValueError(f'Invalid reasoning score: {qid}')
                scores[qid]=score
        if not set(scores) <= set(lookup):
            raise ValueError('Reasoning scores have no matching saved records')
    rows=[]
    for q in manifest['queries']:
        row={**q,**responses.get(q['query_id'],{}),**lookup.get(q['query_id'],{})}
        if q['query_id'] not in lookup:
            row['status']='api_error' if row.get('exception') else 'pending' if q['query_id'] in requested else 'unstarted'
            raw=row.get('raw_response') or {}
            row['assistant_text']=(raw.get('choices') or [{}])[0].get('message',{}).get('content')
        score=scores.get(q['query_id'])
        if score:
            for key in ('H','shots','history_start','query_index','repeat_id','episode_path'):
                if str(row[key]) != score[key]:
                    raise ValueError(f'Reasoning metadata mismatch: {key}')
            if (score['model'] != row['model_alias']
                    or int(score['reward_present']) != int(row['reward_present'])
                    or not score['Reasoning'].strip()
                    or score['Reasoning'].strip() not in (row.get('assistant_text') or '')):
                raise ValueError('Reasoning score does not match saved answer')
            row['reasoning_score']=int(score['score'])
        rows.append(row)
    data=dict(created_at_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(timespec='seconds'),rows=rows)
    template=(HERE/'templates/results_viewer_template.html').read_text()
    payload=json.dumps(data,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    out=HERE/'view_rawdata.html'
    out.write_text(template.replace('__DATASET__',payload))
    print(f'{len(rows)} queries / {len(records)} saved records: {out}')


if __name__=='__main__':
    main()
