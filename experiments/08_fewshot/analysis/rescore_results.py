"""Rescore saved responses offline, preserving original files in a dated backup."""
from importlib import import_module
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments import common
from experiments.split_execution import execution_lock
from llm_x.data import Episode

scoring = import_module('experiments.08_fewshot.runner.scoring')


def rescore(root=HERE / 'results'):
    root = Path(root)
    with execution_lock(root):
        directory = root / 'api'
        records = common.read_jsonl(directory / 'records.jsonl')
        if records and all(r.get('scoring_version') == scoring.VERSION for r in records):
            print('All records already use', scoring.VERSION)
            return False
        manifest = common.read_json(directory / 'manifest.json')
        queries = {q['query_id']: q for q in manifest['queries']}
        responses = {r['query_id']: r for r in common.read_jsonl(directory / 'responses.jsonl')}
        assert len({r['query_id'] for r in records}) == len(records)
        protected = [directory / name for name in ('requests.jsonl', 'responses.jsonl', 'manifest.json')]
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
        episodes, updated = {}, []
        for row in records:
            query = queries[row['query_id']]
            assert all(row[k] == query[k] for k in query)
            raw = responses[row['query_id']]['raw_response']
            text = (raw.get('choices') or [{}])[0].get('message', {}).get('content')
            assert row['raw_response'] == raw and row['assistant_text'] == text
            path = ROOT / row['episode_path']
            if path not in episodes:
                episodes[path] = Episode.load(path)
            assert episodes[path].sha256 == row['episode_sha256']
            evaluation = scoring.score(row, text, episodes[path])
            assert evaluation['ground_truth'] == row['ground_truth']
            result = {k: v for k, v in row.items() if k not in scoring.EVALUATION_FIELDS}
            result['legacy_evaluation'] = row.get('legacy_evaluation', {k: row[k] for k in scoring.EVALUATION_FIELDS if k in row})
            result.update(evaluation)
            updated.append(result)
        summary = common.read_json(directory / 'summary.json')
        summary.update(matches=sum(r['status'] == 'match' for r in updated),
                       mismatches=sum(r['status'] == 'mismatch' for r in updated),
                       ignored=sum(r['status'] == 'ignored' for r in updated),
                       empty_responses=sum(r['status'] == 'empty_response' for r in updated),
                       scoring_version=scoring.VERSION, rescored_at_utc=common.utc_now())
        with tempfile.TemporaryDirectory(prefix='.rescore_', dir=root) as tmp:
            staging = Path(tmp)
            (staging / 'records.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in updated))
            common.write_csv(staging / 'records.csv', [{k: r.get(k) for k in scoring.CSV_FIELDS} for r in updated])
            common.write_json(staging / 'summary.json', summary)
            backup = root / 'scoring_backups' / common.utc_now().replace(':', '-').replace('/', '-')
            backup.mkdir(parents=True, exist_ok=False)
            for name in ('records.jsonl', 'records.csv', 'summary.json'):
                shutil.copy2(directory / name, backup / name)
            common.write_json(backup / 'migration.json', dict(scoring_version=scoring.VERSION,
                              protected_sha256=hashes, records=len(updated),
                              changed_status=sum(a['status'] != b['status'] for a, b in zip(records, updated))))
            for name in ('records.jsonl', 'records.csv', 'summary.json'):
                (staging / name).replace(directory / name)
        assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[p.name] for p in protected)
        print(f'Rescored {len(updated)} records; backup: {backup}')
        print(f"match={summary['matches']}, mismatch={summary['mismatches']}, ignored={summary['ignored']}")
        return True


if __name__ == '__main__':
    rescore()
