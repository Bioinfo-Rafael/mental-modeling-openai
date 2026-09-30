"""Build offline few-shot data and viewer from validated Joint results. No API calls."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
from importlib import import_module
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from experiments.common_analysis.joint_data import load_and_parse, required_components, DIMENSIONS

VIEWER = ROOT / 'experiments/07_view_rawdata'
SCORE = VIEWER / 'reasoning_for_scoring_scored_astra_high.csv'
SOURCE_03 = ROOT / 'experiments/03_1_gpt35_history_n30_joint/results'
SOURCE_06 = ROOT / 'experiments/06_new_models_n10/results/merged/c189af3036db4adfb9e4da5bb632ea48'
DIRECTIONS = ['DEC', 'INC', 'UNCH']
LABELS = {'action': '行動ID', 'action_value': 'トルク', 'action_bin': 'bin ID',
          'direction': '増減方向', 'state_value': '状態値', 'state_delta': '状態変化量（後 − 前）'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparisons(parsed):
    result = {}
    for component in required_components(parsed['task'], parsed['metric']):
        item = parsed['components'][component]
        truth = parsed['gt_' + component]
        prediction = item['value'] if item['ok'] else None
        discrete = component in {'action', 'action_bin', 'direction'}
        matches = [p == t for p, t in zip(prediction, truth)] if item['ok'] and discrete else None
        error = [abs(p-t) for p, t in zip(prediction, truth)] if item['ok'] and component != 'direction' else None
        if component == 'direction':
            truth = [DIRECTIONS[t] for t in truth]
            prediction = [DIRECTIONS[p] for p in prediction] if prediction is not None else None
        result[component] = dict(label=LABELS[component], dimensions=list(DIMENSIONS[parsed['task']])
                                if component in {'direction', 'state_value', 'state_delta'} else [LABELS[component]],
                                prediction=prediction, ground_truth=truth, parsed=item['ok'],
                                parse_error=item['error'], exact_match=all(matches) if matches is not None else None,
                                element_matches=matches, absolute_error=error,
                                mean_absolute_error=sum(error)/len(error) if error is not None else None)
    return result


def build(source03=SOURCE_03, source06=SOURCE_06):
    raw, parsed, hashes = [], {}, {}
    for name, source in [('03_1_gpt35_history_n30_joint', source03), ('06_new_models_n10', source06)]:
        source = source.resolve()
        spec = import_module('experiments.' + name + '.run').EXPERIMENT
        _, _, data, episode_hashes = load_and_parse(source, spec, allow_api_failures=True)
        parsed.update({r['query_id']: r for r in data})
        rows = [json.loads(line) for line in (source / 'records.jsonl').read_text().splitlines() if line.strip()]
        raw.extend(rows)
        hashes.update(episode_hashes)
        for path in (source / 'records.jsonl', source / 'manifest.json', source / 'summary.json'):
            hashes[str(path.relative_to(ROOT))] = sha(path)
    if len(raw) != 1920 or len(parsed) != 1920:
        raise ValueError('Expected 1920 unique queries')
    hashes[str(SCORE.relative_to(ROOT))] = sha(SCORE)
    scores = list(csv.DictReader(SCORE.open(encoding='utf-8-sig')))
    original_scores = list(csv.DictReader((VIEWER / 'reasoning_for_scoring.csv').open(encoding='utf-8-sig')))
    if len(scores) != len(raw) or len(original_scores) != len(scores):
        raise ValueError('Score row count mismatch')
    # The original score export has a documented stable model/task/metric/H/ordinal order.
    rank = {'sol': 0, 'terra': 1, 'luna': 2, '3.5': 3}
    ordered = sorted(raw, key=lambda r: (rank[r['model_alias']], r['task'], r['metric'], r['H'], r['ordinal']))
    counts = Counter()
    scored = {}
    for index, (score, original, record) in enumerate(zip(scores, original_scores, ordered), 1):
        fields = ('model', 'Task', 'Metrics', 'H', 'Reasoning')
        if any(score[k] != original[k] for k in fields):
            raise ValueError(f'Score CSV reordered/edited at record {index}')
        key = (score['model'], score['Task'], score['Metrics'], int(score['H']))
        ordinal = counts[key]; counts[key] += 1
        if (*key, ordinal) != (record['model_alias'], record['task'], record['metric'], record['H'], record['ordinal']):
            raise ValueError(f'Score identity mismatch at record {index}')
        if score['Reasoning'] and score['Reasoning'] not in (record.get('assistant_text') or ''):
            raise ValueError(f'Reasoning mismatch at record {index}')
        value = int(score['score'])
        if value not in (1, 2, 3, 4):
            raise ValueError('Invalid reasoning score')
        scored[record['query_id']] = dict(reasoning_score=value, reasoning=score['Reasoning'],
                                         reasoning_available=bool(score['Reasoning']), score_csv_record=index)
    hashes[str((VIEWER / 'reasoning_for_scoring.csv').relative_to(ROOT))] = sha(VIEWER / 'reasoning_for_scoring.csv')
    examples, viewer = [], []
    for record in ordered:
        qid = record['query_id']; p = parsed[qid]; s = scored[qid]
        compare = comparisons(p)
        example = {k: record[k] for k in ('query_id', 'task', 'metric', 'H', 'ordinal', 'query_index', 'episode_path', 'episode_sha256')}
        example.update(model=record['model_alias'], model_id=record['model'], example_type=None, note='',
                       question=record['user_prompt'], answer=record.get('assistant_text'),
                       **s, ground_truth={k: v['ground_truth'] for k,v in compare.items()},
                       comparison=compare, response_available=bool(record.get('assistant_text')),
                       response_sha256=p['response_sha256'],
                       source_records=p['source_records'], source_line=p['source_line'],
                       score_source=str(SCORE.relative_to(ROOT)), source_experiment=record.get('source_experiment'),
                       reused=record.get('reused', False))
        # system_prompt deliberately absent from the reusable example/export schema.
        examples.append(example)
        viewer.append({**record, 'fewshot': example})
    validation = dict(schema_version=1, records=len(examples), unique_query_ids=len(parsed),
                      by_model=dict(Counter(e['model'] for e in examples)),
                      missing_reasoning=sum(not e['reasoning_available'] for e in examples),
                      missing_responses=sum(not e['response_available'] for e in examples),
                      score_join='CSV stable order + original CSV equality + nonblank reasoning text verified',
                      source_sha256=hashes, direction_threshold=0.0001,
                      state_delta='later - earlier for both next-state and last-state',
                      continuous_correctness='No arbitrary tolerance; absolute errors only')
    for path, expected in hashes.items():
        if sha(ROOT / path) != expected:
            raise ValueError(f'Input changed while building: {path}')
    return examples, viewer, validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source03', type=Path, default=SOURCE_03)
    parser.add_argument('--source06', type=Path, default=SOURCE_06)
    args = parser.parse_args()
    examples, viewer, validation = build(args.source03, args.source06)
    (HERE / 'examples.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    (HERE / 'validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2) + '\n')
    template = (VIEWER / 'viewer_template.html').read_text()
    dataset = json.dumps(viewer, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    html = template.replace('/*VIEWER_CSS*/', (VIEWER / 'viewer.css').read_text())
    html = html.replace('/*VIEWER_JS*/', (VIEWER / 'viewer.js').read_text())
    html = html.replace('__DATASET__', dataset)
    (VIEWER / 'view_rawdata.html').write_text(html)
    print(json.dumps({k:v for k,v in validation.items() if k != 'source_sha256'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
