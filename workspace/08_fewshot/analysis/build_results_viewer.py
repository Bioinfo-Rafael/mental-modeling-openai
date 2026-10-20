"""Build a self-contained offline viewer from a snapshot of the API run."""
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
    rows=[]
    for q in manifest['queries']:
        row={**q,**responses.get(q['query_id'],{}),**lookup.get(q['query_id'],{})}
        if q['query_id'] not in lookup:
            row['status']='api_error' if row.get('exception') else 'pending' if q['query_id'] in requested else 'unstarted'
            raw=row.get('raw_response') or {}
            row['assistant_text']=(raw.get('choices') or [{}])[0].get('message',{}).get('content')
        rows.append(row)
    data=dict(created_at_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(timespec='seconds'),rows=rows)
    template=(HERE/'templates/results_viewer_template.html').read_text()
    payload=json.dumps(data,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    out=HERE/'view_rawdata.html'
    out.write_text(template.replace('__DATASET__',payload))
    print(f'{len(rows)} queries / {len(records)} saved records: {out}')


if __name__=='__main__':
    main()
