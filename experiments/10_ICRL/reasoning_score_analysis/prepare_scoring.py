"""Prepare unscored public Reasoning using Exp.09's extractor; no API calls."""
import csv
import hashlib
from importlib import import_module
from itertools import product
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
previous = import_module('experiments.09_ablation_reward.reasoning_score_analysis.prepare_scoring')
extract = previous.extract
ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('query_id', 'condition_id', 'model', 'Task', 'Metrics', 'H', 'method', 'K', 'E',
          'ordinal', 'history_start', 'query_index', 'repeat_id', 'reward_present',
          'target_episode', 'episode_path', 'Reasoning', 'score')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root=ROOT):
    source = root / 'results/api/records.jsonl'
    manifest_path = root / 'results/api/manifest.json'
    destination = root / 'reasoning_for_scoring.csv'
    validation = root / 'reasoning_score_analysis/validation.json'
    scored = root / 'reasoning_for_scoring_scored_astra_high.csv'
    for path in (destination, validation, scored):
        if path.exists():
            raise FileExistsError(f'Existing scoring artifact; will not overwrite: {path}')
    source_hash, manifest_hash = sha256(source), sha256(manifest_path)
    records = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
    queries = json.loads(manifest_path.read_text(encoding='utf-8'))['queries']
    ids = [r['query_id'] for r in records]
    lookup = {q['query_id']: q for q in queries}
    if len(records) != 72 or len(set(ids)) != 72 or len(queries) != 72 or set(ids) != set(lookup):
        raise ValueError('Expected all 72 unique planned responses')
    expected = set(product(('terra', 'luna'), ('direct', 'ppo_prior', 'icrfqi_k2', 'icrfqi_k5'),
                           (1, 3, 9), (1, 2, 3)))
    if {(r['model_alias'], r['method'], r['E'], r['repeat_id']) for r in records} != expected:
        raise ValueError('Unexpected model/method/E/repeat grid')
    rows = []
    for record in records:
        query = lookup[record['query_id']]
        # Join only matching metadata; keep the original response order.
        keys = ('model_alias', 'task', 'metric', 'context_episode_count') + tuple(
            k for k in FIELDS if k not in ('model', 'Task', 'Metrics', 'Reasoning', 'score'))
        if any(record[k] != query[k] for k in keys):
            raise ValueError(f"Metadata differs from manifest: {record['query_id']}")
        text = record['assistant_text']
        if not isinstance(text, str) or text != record['raw_response']['choices'][0]['message']['content']:
            raise ValueError(f"Saved answer differs from raw response: {record['query_id']}")
        row = {key: record[key] for key in FIELDS if key in record}
        row.update(model=record['model_alias'], Task=record['task'], Metrics=record['metric'],
                   reward_present=int(record['reward_present']), Reasoning=extract(text), score='')
        rows.append(row)
    # Finish validation and extraction for every answer before creating output.
    if (sha256(source), sha256(manifest_path)) != (source_hash, manifest_hash):
        raise ValueError('Inputs changed during extraction')
    with destination.open('x', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    with destination.open(encoding='utf-8-sig', newline='') as handle:
        saved = list(csv.DictReader(handle))
    expected_rows = [{k: '' if r[k] is None else str(r[k]) for k in FIELDS} for r in rows]
    if saved != expected_rows:
        raise ValueError('CSV roundtrip changed content or row order')
    report = dict(source='results/api/records.jsonl', source_sha256=source_hash,
                  manifest='results/api/manifest.json', manifest_sha256=manifest_hash,
                  extractor='experiments/09_ablation_reward/reasoning_score_analysis/prepare_scoring.py',
                  extractor_sha256=sha256(Path(previous.__file__)),
                  rows=len(saved), unique_query_ids=len(set(ids)), manifest_ids_match=True,
                  metadata_matches_manifest=True, raw_response_text_verified=True,
                  extracted_reasoning=len(saved), empty_reasoning=0, empty_scores=len(saved),
                  row_order_preserved=True, csv_roundtrip_verified=True,
                  output_sha256=sha256(destination), score_scale='1=generic, 2=policy, 3=specific, 4=quantitative')
    validation.parent.mkdir(parents=True, exist_ok=True)
    with validation.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(report, indent=2) + '\n')
    print(f'Prepared {len(saved)} unscored Reasoning rows: {destination}')
    return report


if __name__ == '__main__':
    prepare()
