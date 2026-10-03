"""Condition metrics and reward-on/off differences, including missing results."""
import sys
from pathlib import Path
from collections import defaultdict
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments import common

HERE = Path(__file__).resolve().parents[1]


def summarize(root):
    manifest = common.read_json(root / 'api/manifest.json')
    records_path = root / 'api/records.jsonl'
    records = common.read_jsonl(records_path) if records_path.exists() else []
    request_path = root / 'api/requests.jsonl'
    attempted = {r['query_id'] for r in common.read_jsonl(request_path)} if request_path.exists() else set()
    lookup = {r['query_id']:r for r in records}
    groups = defaultdict(list)
    for q in manifest['queries']:
        groups[q['condition_id']].append(q)
    metrics = []
    for condition, queries in groups.items():
        result = [lookup[q['query_id']] for q in queries if q['query_id'] in lookup]
        valid = [r for r in result if r['status'] in ('match', 'mismatch')]
        errors = [r['absolute_error'] for r in result if r.get('absolute_error') is not None]
        matches = sum(r['status']=='match' for r in valid)
        q = queries[0]
        metrics.append(dict(condition_id=condition, **{k:q[k] for k in
                       ('model_alias','H','history_start','shots','reward_present')},
                       planned=len(queries), responses=len(result), parsed_bins=len(valid),
                       matches=matches, bin_accuracy=matches/len(valid) if valid else None,
                       match_rate_all_planned=matches/len(queries),
                       action_mae=mean(errors) if errors else None, action_mae_n=len(errors),
                       parse_or_empty=len(result)-len(valid),
                       unattempted=sum(q['query_id'] not in attempted for q in queries),
                       unresolved=sum(q['query_id'] in attempted and q['query_id'] not in lookup for q in queries)))
    pairs = defaultdict(dict)
    for row in metrics:
        pairs[tuple(row[k] for k in ('model_alias','H','history_start','shots'))][row['reward_present']] = row
    differences = []
    for key, pair in pairs.items():
        on, off = pair.get(True), pair.get(False)
        if on is None or off is None:
            continue
        row = dict(zip(('model_alias','H','history_start','shots'), key))
        complete = all(r['responses']==r['planned'] for r in (on,off))
        row['all_responses_received'] = complete
        for field, count in (('bin_accuracy','parsed_bins'), ('action_mae','action_mae_n')):
            # Do not compare different surviving subsets if either side has missing/invalid data.
            ready = complete and all(r[count]==r['planned'] for r in (on,off))
            row[f'{field}_off_minus_on'] = off[field]-on[field] if ready else None
        differences.append(row)
    output = root / 'analysis'
    output.mkdir(exist_ok=True)
    common.write_json(output / 'summary.json', dict(conditions=metrics, reward_differences=differences,
        note='Accuracy denominator: parsed bins; MAE denominator: parsed numeric actions. Differences require all repeats valid on both sides. Repeats are not distinct evaluation examples.'))
    common.write_csv(output / 'conditions.csv', metrics)
    if differences:
        common.write_csv(output / 'reward_differences.csv', differences)
    return metrics, differences


if __name__ == '__main__':
    summarize(HERE / 'results')
