"""Read-only joins shared by the ICRL plots and offline HTML viewer."""
from collections import Counter
import csv
import hashlib
from importlib import import_module
from itertools import product
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
MODELS = ('terra', 'luna')
METHODS = ('direct', 'ppo_prior', 'icrfqi_k2', 'icrfqi_k5')
METHOD_LABELS = ('Direct', 'PPO', 'ICRL (K=2)', 'ICRL (K=5)')
EPISODES = (1, 3, 9)
prepare = import_module('experiments.10_ICRL.reasoning_score_analysis.prepare_scoring')
read_journal = import_module('experiments.09_ablation_reward.analysis.build_results_viewer').read_journal


def require(condition, message):
    if not condition:
        raise ValueError(message)


def index_rows(rows, label):
    result = {r['query_id']: r for r in rows}
    require(len(result) == len(rows), f'Duplicate query_id in {label}')
    return result


def validate_metrics(row):
    """Check saved Joint metrics without introducing another response parser."""
    for key in ('prediction', 'ground_truth'):
        values = row.get(key)
        require(isinstance(values, list) and len(values) == 1
                and type(values[0]) is int and 0 <= values[0] <= 9, f'Invalid action bin: {key}')
    require((row['status'] == 'match') == (row['prediction'] == row['ground_truth']),
            'Saved bin status disagrees with prediction/truth')
    for key in ('action_value_prediction', 'action_value_ground_truth'):
        values = row.get(key)
        require(isinstance(values, list) and len(values) == 1
                and type(values[0]) in (int, float) and math.isfinite(values[0]),
                f'Missing or invalid numeric action: {key}')
    error = row.get('absolute_error')
    require(type(error) in (int, float) and math.isfinite(error) and error >= 0,
            'Missing or invalid action error')
    require(math.isclose(error, abs(row['action_value_prediction'][0] - row['action_value_ground_truth'][0]),
                         rel_tol=0, abs_tol=1e-12), 'Saved absolute error disagrees with actions')


def load_rows(root=HERE, *, require_complete=False):
    source = root / 'results/api'
    manifest_path = source / 'manifest.json'
    score_path = root / 'reasoning_for_scoring_scored_astra_high.csv'
    paths = [manifest_path, *(source / name for name in ('records.jsonl', 'responses.jsonl', 'requests.jsonl')),
             score_path]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in paths if p.exists()}
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    queries = manifest['queries']
    planned = index_rows(queries, 'manifest')
    grid = Counter((q['model_alias'], q['method'], q['E'], q['repeat_id']) for q in queries)
    expected = set(product(MODELS, METHODS, EPISODES, (1, 2, 3)))
    require(len(queries) == 72 and set(grid) == expected and set(grid.values()) == {1},
            'Expected the complete 72-query ICRL grid')
    expected_k = {'direct': None, 'ppo_prior': None, 'icrfqi_k2': 2, 'icrfqi_k5': 5}
    for q in queries:
        require((q['task'], q['metric'], q['H'], q['history_start'], q['query_index']) ==
                ('Pendulum-v1', 'next-action', 20, 0, 20), 'Unexpected target window')
        require(q['target_episode'] == Path(q['episode_path']).name == 'episode_9_seed3407.npz'
                and q['reward_present'] is True and q['context_episode_count'] == q['E']
                and q['K'] == expected_k[q['method']] and q['ordinal'] == q['repeat_id'] - 1,
                'Unexpected episode or condition metadata')
    records = index_rows(read_journal(source / 'records.jsonl'), 'records')
    responses = index_rows(read_journal(source / 'responses.jsonl'), 'responses')
    requested = index_rows(read_journal(source / 'requests.jsonl'), 'requests')
    require(set(records) <= set(planned) and set(responses) <= set(requested) <= set(planned),
            'Unexpected query IDs in API journals')
    require(set(records) <= set(responses), 'Saved record without a response journal')
    scores = {}
    if score_path.exists():
        with score_path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            require(reader.fieldnames == list(prepare.FIELDS), 'Reasoning CSV columns changed')
            scores = index_rows(list(reader), 'reasoning CSV')
        require(set(scores) <= set(records), 'Reasoning score without a saved record')
    rows = []
    for q in queries:
        qid = q['query_id']
        record = records.get(qid)
        if record is not None:
            require(all(record.get(k) == v for k, v in q.items()), f'Record differs from manifest: {qid}')
            require(record.get('raw_response') == responses[qid].get('raw_response'),
                    f'Record differs from response journal: {qid}')
            raw = record.get('raw_response') or {}
            text = (raw.get('choices') or [{}])[0].get('message', {}).get('content')
            require(record.get('assistant_text') == text, f'Answer differs from raw response: {qid}')
            row = dict(record)
        else:
            row = {**q, **responses.get(qid, {})}
            row['status'] = ('api_error' if row.get('exception') else
                             'pending' if qid in requested else 'unstarted')
            raw = row.get('raw_response') or {}
            row['assistant_text'] = (raw.get('choices') or [{}])[0].get('message', {}).get('content')
        # Missing or blank manual scores remain missing, never zero.
        row['reasoning_score'] = None
        if qid in scores:
            score = scores[qid]
            expected_csv = {k: row[k] for k in prepare.FIELDS if k in row}
            expected_csv.update(model=row['model_alias'], Task=row['task'], Metrics=row['metric'],
                                reward_present=int(row['reward_present']),
                                Reasoning=prepare.extract(row['assistant_text']))
            for k in prepare.FIELDS:
                if k != 'score':
                    value = '' if expected_csv[k] is None else str(expected_csv[k])
                    require(score[k] == value, f'Reasoning CSV differs from answer metadata/body: {qid} / {k}')
            require(score['score'] in ('', '1', '2', '3', '4'), f'Invalid reasoning score: {qid}')
            row['reasoning_score'] = int(score['score']) if score['score'] else None
        if require_complete:
            require(record is not None and row['status'] in ('match', 'mismatch'),
                    'Plotting requires all 72 responses with parsed action bins')
            require(row['reasoning_score'] is not None,
                    'Plotting requires all 72 reasoning scores (1–4) in reasoning_for_scoring_scored_astra_high.csv')
            validate_metrics(row)
        rows.append(row)
    require(all(hashlib.sha256((root / p).read_bytes()).hexdigest() == digest for p, digest in hashes.items()),
            'Source files changed while reading; retry after saving is complete')
    return rows, hashes


def load_samples(root=HERE):
    rows, hashes = load_rows(root, require_complete=True)
    samples = [dict(query_id=r['query_id'], model=r['model_alias'], method=r['method'], K=r['K'],
                    E=r['E'], repeat_id=r['repeat_id'], H=r['H'], history_start=r['history_start'],
                    query_index=r['query_index'], episode_path=r['episode_path'],
                    correct=int(r['status'] == 'match'), absolute_error=r['absolute_error'],
                    reasoning_score=r['reasoning_score']) for r in rows]
    return samples, hashes
