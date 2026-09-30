"""Pool two experimental factors and compare the third, without API calls."""
import csv
import hashlib
import itertools
from importlib import import_module
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter, MaxNLocator
import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.common_analysis.joint_data import Episode
scoring = import_module('experiments.08_fewshot.runner.scoring')

TASKS = ('MountainCar-v0', 'Pendulum-v1')
MODELS = ('terra', 'luna')
COLORS = {'terra': '#2468a2', 'luna': '#cf6638'}
FACTORS = {
    'score_pattern': ('Score pattern', ('pattern1', 'pattern2'), ('Pattern 1', 'Pattern 2'), 'shots + H'),
    'shots': ('Few-shot total', (4, 8, 12), ('4', '8', '12'), 'score pattern + H'),
    'H': ('History length H', (5, 20), ('5', '20'), 'score pattern + shots'),
}
OUTPUT = HERE / 'results/analysis/pooled_comparisons'


def load_records():
    source = HERE / 'results/api/records.jsonl'
    score_path = HERE / 'reasoning_for_scoring_scored_astra_high.csv'
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    with score_path.open(encoding='utf-8-sig', newline='') as f:
        scores = list(csv.DictReader(f))
    by_id = {r['query_id']: r for r in scores}
    assert len(by_id) == len(scores) == len(rows) == 144
    assert len({r['query_id'] for r in rows}) == 144
    assert set(by_id) == {r['query_id'] for r in rows}
    expected = set(itertools.product(TASKS, MODELS, ('pattern1', 'pattern2'), (4, 8, 12), (5, 20), range(3)))
    assert {(r['task'], r['model_alias'], r['score_pattern'], r['shots'], r['H'], r['ordinal']) for r in rows} == expected
    episodes, records = {}, []
    for row in rows:
        score = by_id[row['query_id']]
        for column, key in [('model', 'model_alias'), ('Task', 'task'), ('Metrics', 'metric'),
                            ('H', 'H'), ('shots', 'shots'), ('score_pattern', 'score_pattern'), ('ordinal', 'ordinal')]:
            assert score[column] == str(row[key]), (row['query_id'], column)
        assert row['metric'] == 'next-action'
        path = ROOT / row['episode_path']
        if path not in episodes:
            episodes[path] = Episode.load(path)
        episode = episodes[path]
        assert episode.sha256 == row['episode_sha256']
        assert row['index'] == row['query_index']
        truth = episode.action_vector(row['query_index']).astype(float).tolist()
        evaluation = scoring.score(row, row['assistant_text'], episode)
        assert row.get('scoring_version') == scoring.VERSION, 'Run analysis/rescore_results.py first'
        assert row['status'] == evaluation['status'] and row['prediction'] == evaluation['prediction']
        mountain = row['task'] == TASKS[0]
        gt_id = [evaluation['ground_truth']] if mountain else evaluation['ground_truth']
        prediction = evaluation['prediction']
        discrete = {'ok': prediction is not None,
                    'value': [prediction] if mountain else prediction}
        numeric = {'ok': evaluation['action_value_prediction'] is not None,
                   'value': evaluation['action_value_prediction']}
        saved_gt = [row['ground_truth']] if row['task'] == TASKS[0] else row['ground_truth']
        assert gt_id == saved_gt
        assert score['score'] in {'1', '2', '3', '4'}
        records.append({k: row[k] for k in ('query_id', 'task', 'model_alias', 'score_pattern', 'shots', 'H', 'ordinal', 'episode_path', 'query_index')} | {
            'correct': int(discrete['ok'] and discrete['value'] == gt_id),
            'accuracy_parsed': discrete['ok'], 'error_parsed': numeric['ok'],
            'predicted_id': discrete['value'][0] if discrete['ok'] else None,
            'true_id': gt_id[0],
            'predicted_value': numeric['value'][0] if numeric['ok'] else None,
            'true_value': truth[0],
            'absolute_error': abs(numeric['value'][0] - truth[0]) if numeric['ok'] else None,
            'reasoning_score': int(score['score']),
            'legacy_correct': int(row.get('legacy_evaluation', row)['status'] == 'match'),
        })
    # The same three physical questions must be shared by every condition.
    for task in TASKS:
        task_rows = [r for r in records if r['task'] == task]
        assert len({(r['episode_path'], r['query_index']) for r in task_rows}) == 3
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (source, score_path, *episodes)}
    return records, hashes


def aggregate(records):
    output = []
    for task, (factor, (_, levels, _, _)), model in itertools.product(TASKS, FACTORS.items(), MODELS):
        for level in levels:
            group = [r for r in records if r['task'] == task and r['model_alias'] == model and r[factor] == level]
            assert len(group) == (12 if factor == 'shots' else 18)
            errors = [r['absolute_error'] for r in group if r['absolute_error'] is not None]
            output.append(dict(task=task, factor=factor, level=level, model=model, n=len(group),
                               correct=sum(r['correct'] for r in group),
                               accuracy=np.mean([r['correct'] for r in group]).item(),
                               error_n=len(errors), mae=float(np.mean(errors)) if errors else None,
                               score_n=len(group), reasoning_score=float(np.mean([r['reasoning_score'] for r in group])),
                               accuracy_unparsed=sum(not r['accuracy_parsed'] for r in group)))
    return output


def panel(ax, task, factor, metric, summaries):
    title, levels, labels, pooled = FACTORS[factor]
    x = np.arange(len(levels))
    line = factor == 'shots'
    for model_index, model in enumerate(MODELS):
        group = [next(r for r in summaries if (r['task'], r['factor'], r['level'], r['model']) == (task, factor, level, model)) for level in levels]
        values = [r[metric] for r in group]
        positions = x if line else x + (model_index - .5) * .32
        if line:
            ax.plot(positions, values, color=COLORS[model], marker='o' if model == 'terra' else 's',
                    linestyle='-' if model == 'terra' else '--', linewidth=2.2, markersize=7,
                    markerfacecolor='white' if model == 'luna' else COLORS[model], label=model.title())
        else:
            ax.bar(positions, values, width=.29, color=COLORS[model], label=model.title(), zorder=3)
        for level, position, value in zip(levels, positions, values):
            label = f'{value:.1%}' if metric == 'accuracy' else f'{value:.3f}' if metric == 'mae' else f'{value:.2f}'
            peer = next(r[metric] for r in summaries if (r['task'], r['factor'], r['level'], r['model']) == (task, factor, level, MODELS[1-model_index]))
            above = value > peer or (value == peer and model_index == 0)
            offset = (10 if above else -20) if line else 5
            ax.annotate(label, (position, value), xytext=(0, offset), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9, color=COLORS[model] if line else '#273447')
    ax.set_xticks(x, labels)
    ax.set_xlabel(f'{title}  |  pooled: {pooled}', fontsize=9, labelpad=10)
    ax.grid(axis='y', color='#e4e8ef', zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-.55, len(levels) - .45)
    if metric == 'accuracy':
        ax.set_ylim(-.12 if line else 0, 1.18)
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title('Action accuracy (higher is better)', fontsize=11)
    elif metric == 'mae':
        maximum = max(r['mae'] for r in summaries if r['task'] == task)
        top = max(.05, maximum * 1.4)
        ax.set_ylim(-.14 * top if line else 0, top)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        ax.set_title('MAE: action ID (lower is better)' if task == TASKS[0] else 'MAE: torque (lower is better)', fontsize=11)
    else:
        ax.set_ylim(0, 4.4)
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_title('Mean reasoning score (1–4; higher is better)', fontsize=11)


def save_figure(task, factors, summaries, stem):
    fig, axes = plt.subplots(len(factors), 3, figsize=(16, 4.1 * len(factors) + 1.1), squeeze=False)
    for i, factor in enumerate(factors):
        for j, metric in enumerate(('accuracy', 'mae', 'reasoning_score')):
            panel(axes[i, j], task, factor, metric, summaries)
    fig.suptitle(f'{task} | Pooled comparisons', fontsize=21, y=.985)
    handles = [Line2D([0], [0], color=COLORS[m], marker='o' if m == 'terra' else 's', lw=3, label=m.title()) for m in MODELS]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .954), ncol=2, frameon=False)
    foot = ('Per model/group: pattern or H = 18 responses; shots = 12 responses. Same 3 target questions across conditions.\n'
            'Accuracy: final action ID / bin ID. MAE: parsed numeric predictions. Reasoning: newly scored 1–4 scale.\n'
            'Pattern labels retain the original example-score scale. Descriptive pooled means; no independence-based error bars.')
    fig.text(.055, .016, foot, fontsize=9, color='#586273', linespacing=1.5)
    fig.tight_layout(rect=(.015, .11 if len(factors) == 1 else .065, .995, .89 if len(factors) == 1 else .94), h_pad=3)
    for extension in ('png',):
        fig.savefig(OUTPUT / f'{stem}.{extension}', dpi=180, facecolor='white')
    plt.close(fig)


def main():
    records, hashes = load_records()
    summaries = aggregate(records)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    for filename, rows in [('pooled_metrics.csv', summaries), ('per_query_metrics.csv', records)]:
        with (OUTPUT / filename).open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    for task in TASKS:
        save_figure(task, tuple(FACTORS), summaries, f'{task}_overview')
        for factor in FACTORS:
            save_figure(task, (factor,), summaries, f'{task}_by_{factor}')
    audit = dict(source_sha256=hashes, records=len(records), groups=len(summaries),
                 accuracy_unparsed=sum(not r['accuracy_parsed'] for r in records),
                 error_unparsed=sum(not r['error_parsed'] for r in records),
                 corrected_status_count=sum(r['correct'] != r['legacy_correct'] for r in records),
                 totals={task: {'correct': sum(r['correct'] for r in records if r['task'] == task),
                                'legacy_correct': sum(r['legacy_correct'] for r in records if r['task'] == task)} for task in TASKS})
    (OUTPUT / 'validation.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == value for p, value in hashes.items())
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(OUTPUT)


if __name__ == '__main__':
    main()
