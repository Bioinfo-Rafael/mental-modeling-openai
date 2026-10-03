"""Plot the four requested reward/timepoint comparisons from saved results only."""
from collections import Counter, defaultdict
import csv
import hashlib
from itertools import product
import json
import math
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / 'results/analysis/comparisons'
MODELS = ('terra', 'luna')
COLORS = ('#3973AC', '#D77B35')
TIMEPOINT_COLORS = ('#8B5FBF', '#32936F')
METRICS = (
    ('accuracy_pct', 'Action-bin accuracy', 'Accuracy (%)', '.1f', 'YlGnBu'),
    ('action_mae', 'Action error', 'Mean absolute error (torque)', '.3f', 'YlOrRd'),
    ('reasoning_mean', 'Reasoning level', 'Mean score (1–4)', '.2f', 'YlGnBu'),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_samples():
    paths = [HERE / 'results/api/records.jsonl', HERE / 'results/api/manifest.json',
             HERE / 'reasoning_for_scoring_scored_astra_high.csv']
    records = read_jsonl(paths[0])
    manifest = json.loads(paths[1].read_text())
    with paths[2].open(encoding='utf-8-sig', newline='') as handle:
        scores = list(csv.DictReader(handle))
    ids = [r['query_id'] for r in records]
    score_ids = [r['query_id'] for r in scores]
    planned = [r['query_id'] for r in manifest['queries']]
    require(len(ids) == len(set(ids)) == len(score_ids) == len(set(score_ids)) == 96,
            'Expected 96 unique records and scores')
    require(len(planned) == len(set(planned)) == 96 and set(ids) == set(score_ids) == set(planned),
            'Record, score and manifest query IDs do not match')
    lookup = {r['query_id']: r for r in scores}
    queries = {r['query_id']: r for r in manifest['queries']}
    samples = []
    for r in records:
        s, q = lookup[r['query_id']], queries[r['query_id']]
        for key in ('H', 'shots', 'history_start', 'query_index', 'repeat_id', 'episode_path'):
            require(str(r[key]) == s[key] and r[key] == q[key], f'Metadata mismatch: {key}')
        require(s['model'] == r['model_alias'] == q['model_alias'], 'Model mismatch')
        require(s['Task'] == r['task'] == 'Pendulum-v1' and s['Metrics'] == r['metric'] == 'next-action',
                'Unexpected task or metric')
        require(int(s['reward_present']) == int(r['reward_present']) == int(q['reward_present']),
                'Reward condition mismatch')
        require(r['reward_scope'] == s['reward_scope'] == 'target_history_only', 'Unexpected reward scope')
        require(s['Reasoning'].strip() in r['assistant_text'], 'Scored Reasoning differs from answer')
        require(r['status'] in ('match', 'mismatch'), 'Missing or unparsed bin prediction')
        require(s['score'] in ('1', '2', '3', '4'), 'Missing or invalid reasoning score')
        error = r['absolute_error']
        require(isinstance(error, (float, int)) and math.isfinite(error) and error >= 0, 'Invalid action error')
        require(math.isclose(error, abs(r['action_value_prediction'][0] - r['action_value_ground_truth'][0]),
                             abs_tol=1e-12), 'Action MAE does not match prediction/truth')
        require((r['status'] == 'match') == (r['prediction'] == r['ground_truth']), 'Bin scoring mismatch')
        samples.append(dict(query_id=r['query_id'], model=r['model_alias'], H=r['H'], shots=r['shots'],
                            start=r['history_start'], reward=int(r['reward_present']), repeat=r['repeat_id'],
                            query_index=r['query_index'], episode_path=r['episode_path'],
                            correct=int(r['status'] == 'match'), absolute_error=error,
                            reasoning_score=int(s['score'])))
    grid = Counter(tuple(r[k] for k in ('model', 'H', 'shots', 'start', 'reward', 'repeat')) for r in samples)
    expected = set(product(MODELS, (5, 20), (0, 4), (0, 100), (1, 0), (1, 2, 3)))
    require(set(grid) == expected and set(grid.values()) == {1}, 'Experiment grid is incomplete or duplicated')
    require(all(r['query_index'] == r['start'] + r['H'] for r in samples), 'Unexpected target timepoint')
    return samples, {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def aggregate(samples, keys, expected_n):
    groups = defaultdict(list)
    for row in samples:
        groups[tuple(row[k] for k in keys)].append(row)
    rows = []
    for key, group in sorted(groups.items()):
        require(len(group) == expected_n, f'Expected n={expected_n}, got {len(group)} for {key}')
        rows.append(dict(zip(keys, key), n=len(group), matches=sum(r['correct'] for r in group),
                         accuracy_pct=100 * mean(r['correct'] for r in group),
                         action_mae=mean(r['absolute_error'] for r in group),
                         reasoning_mean=mean(r['reasoning_score'] for r in group)))
    return rows


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def find(rows, **conditions):
    matches = [r for r in rows if all(r[k] == v for k, v in conditions.items())]
    require(len(matches) == 1, f'Ambiguous group: {conditions}')
    return matches[0]


def bar_axis(ax, metric, max_mae):
    key, title, label, _, _ = metric
    ax.set_title(title, fontsize=14, pad=14)
    ax.set_ylabel(label)
    ax.set_ylim(0, {'accuracy_pct': 115, 'action_mae': max_mae * 1.20, 'reasoning_mean': 4.6}[key])
    if key == 'accuracy_pct':
        ax.set_yticks(range(0, 101, 20))
    elif key == 'reasoning_mean':
        ax.set_yticks(range(5))
    ax.grid(axis='y', color='#DFE4EA', linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)


def annotate(ax, bars, rows, metric):
    key, _, _, fmt, _ = metric
    for bar, row in zip(bars, rows):
        value = format(row[key], fmt) + ('%' if key == 'accuracy_pct' else '')
        ax.annotate(f'{value}\nn={row["n"]}', (bar.get_x()+bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points', ha='center', va='bottom', fontsize=10)


def save(fig, name):
    path = OUT / 'figures' / name
    fig.savefig(path, dpi=180, facecolor='white')
    plt.close(fig)
    return str(path.relative_to(OUT))


def grouped_bars(rows, axis, values, labels, title, subtitle, max_mae, name):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.subplots_adjust(left=.06, right=.98, bottom=.14, top=.68, wspace=.30)
    fig.suptitle(title, fontsize=20, y=.98)
    fig.text(.5, .89, subtitle, ha='center', fontsize=11, color='#475569')
    fig.legend([Patch(color=c) for c in COLORS], ['Terra', 'Luna'], loc='upper center',
               bbox_to_anchor=(.5, .855), ncol=2, frameon=False)
    for ax, metric in zip(axes, METRICS):
        bar_axis(ax, metric, max_mae)
        for i, model in enumerate(MODELS):
            group = [find(rows, model=model, **{axis: value}) for value in values]
            bars = ax.bar(np.arange(2) + (i-.5)*.32, [r[metric[0]] for r in group],
                          width=.30, color=COLORS[i])
            annotate(ax, bars, group, metric)
        ax.set_xticks(range(2), labels)
        ax.set_xlim(-.6, 1.6)
    return save(fig, name)


def interaction_bars(rows, max_mae):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.subplots_adjust(left=.07, right=.98, bottom=.10, top=.78, hspace=.65, wspace=.30)
    fig.suptitle('Reward presence × history start', fontsize=21, y=.97)
    fig.text(.5, .92, 'Pooled over H and shots; n=12 per bar. Top: Terra / bottom: Luna.', ha='center')
    fig.legend([Patch(color=c) for c in TIMEPOINT_COLORS], ['t=0', 't=100'], ncol=2,
               loc='upper center', bbox_to_anchor=(.5, .895), frameon=False)
    for i, model in enumerate(MODELS):
        for ax, metric in zip(axes[i], METRICS):
            bar_axis(ax, metric, max_mae)
            ax.set_title(f'{model.title()} · {metric[1]}', fontsize=14, pad=14)
            group = [find(rows, model=model, reward=reward, start=start) for reward, start in
                     ((1, 0), (1, 100), (0, 0), (0, 100))]
            bars = ax.bar([0, 1, 3, 4], [r[metric[0]] for r in group], color=TIMEPOINT_COLORS*2, width=.72)
            annotate(ax, bars, group, metric)
            ax.set_xticks([0, 1, 3, 4], ['t=0', 't=100']*2)
            ax.set_xlim(-.7, 4.7)
            ax.text(.5, -.19, 'Reward present', transform=ax.get_xaxis_transform(), ha='center')
            ax.text(3.5, -.19, 'Reward absent', transform=ax.get_xaxis_transform(), ha='center')
    return save(fig, '03_reward_timepoint_by_model.png')


def heatmaps(rows, scope, max_mae, n):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.subplots_adjust(left=.08, right=.96, bottom=.10, top=.70, wspace=.42)
    title = 'Both models pooled' if scope == 'pooled' else scope.title()
    fig.suptitle(f'{title} · shots / H × reward / history start', fontsize=20, y=.98)
    fig.text(.5, .90, f'n={n} per cell. Numbers are cell means; accuracy is a percentage.', ha='center')
    for ax, metric in zip(axes, METRICS):
        key, label, _, fmt, cmap = metric
        matrix = np.array([[find(rows, shots=shots, H=H, reward=reward, start=start)[key]
                            for reward, start in ((1, 0), (1, 100), (0, 0), (0, 100))]
                          for shots, H in ((0, 5), (0, 20), (4, 5), (4, 20))])
        lo, hi = {'accuracy_pct': (0, 100), 'action_mae': (0, max_mae), 'reasoning_mean': (1, 4)}[key]
        im = ax.imshow(matrix, vmin=lo, vmax=hi, cmap=cmap, aspect='auto')
        ax.set_title(label, fontsize=14, y=1.26)
        ax.set_xticks(range(4), ['t=0', 't=100', 't=0', 't=100'])
        ax.xaxis.tick_top()
        ax.tick_params(length=0, pad=8)
        ax.set_yticks(range(4), ['H=5', 'H=20', 'H=5', 'H=20'])
        for x, text in ((.25, 'Reward present'), (.75, 'Reward absent')):
            ax.text(x, 1.16, text, transform=ax.transAxes, ha='center', fontsize=11, fontweight='bold')
        for y, text in ((.75, '0-shot'), (.25, '4-shot')):
            ax.text(-.24, y, text, transform=ax.transAxes, ha='center', va='center', rotation=90,
                    fontsize=11, fontweight='bold')
        ax.axhline(1.5, color='white', linewidth=4)
        ax.axvline(1.5, color='white', linewidth=4)
        for (i, j), value in np.ndenumerate(matrix):
            text = format(value, fmt) + ('%' if key == 'accuracy_pct' else '')
            ax.text(j, i, text, ha='center', va='center', fontsize=12, fontweight='bold',
                    color='white' if (value-lo)/(hi-lo) > .55 else '#17212B')
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04, shrink=.88)
    number = {'pooled': '04', 'terra': '05', 'luna': '06'}[scope]
    return save(fig, f'{number}_matrix_{scope}.png')


def reward_effects(interaction):
    effects = []
    for model in MODELS:
        row = dict(model=model, n_per_cell=12)
        for metric, *_ in METRICS:
            delta = {}
            for start in (0, 100):
                delta[start] = (find(interaction, model=model, start=start, reward=0)[metric]
                                - find(interaction, model=model, start=start, reward=1)[metric])
                row[f'{metric}_off_minus_on_t{start}'] = delta[start]
            row[f'{metric}_interaction_t100_minus_t0'] = delta[100] - delta[0]
        effects.append(row)
    return effects


def main():
    samples, hashes = load_samples()
    timepoint = aggregate(samples, ('model', 'start'), 24)
    reward = aggregate(samples, ('model', 'reward'), 24)
    interaction = aggregate(samples, ('model', 'reward', 'start'), 12)
    matrix = aggregate(samples, ('shots', 'H', 'reward', 'start'), 6)
    per_model = aggregate(samples, ('model', 'shots', 'H', 'reward', 'start'), 3)
    max_mae = math.ceil(max(r['action_mae'] for r in per_model)*10)/10 or .1
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    tables = {'joined_samples': samples, 'timepoint_summary': timepoint, 'reward_summary': reward,
              'reward_timepoint_summary': interaction,
              'matrix_summary': [dict(model='pooled', **r) for r in matrix] + per_model,
              'reward_effects': reward_effects(interaction)}
    for name, rows in tables.items():
        write_csv(OUT / f'{name}.csv', rows)
    figures = [
        grouped_bars(timepoint, 'start', (0, 100), ('t=0', 't=100'), 'History-start comparison',
                     'Pooled over H, shots and reward presence; n=24 per bar.', max_mae, '01_timepoint_by_model.png'),
        grouped_bars(reward, 'reward', (1, 0), ('Reward present', 'Reward absent'), 'Reward-presence comparison',
                     'Pooled over H, shots and history start; n=24 per bar.', max_mae, '02_reward_by_model.png'),
        interaction_bars(interaction, max_mae),
        heatmaps(matrix, 'pooled', max_mae, 6),
        *[heatmaps([r for r in per_model if r['model'] == model], model, max_mae, 3) for model in MODELS],
    ]
    require(sum(r['matches'] for r in timepoint) == sum(r['correct'] for r in samples), 'Incorrect totals')
    report = dict(samples=len(samples), matches=sum(r['correct'] for r in samples), missing=0,
                  group_counts={'timepoint': len(timepoint), 'reward': len(reward),
                                'interaction': len(interaction), 'matrix_pooled': len(matrix),
                                'matrix_per_model': len(per_model)},
                  n_per_group={'timepoint': 24, 'reward': 24, 'interaction': 12,
                               'matrix_pooled': 6, 'matrix_single_model': 3},
                  reasoning_counts=dict(Counter(r['reasoning_score'] for r in samples)),
                  source_sha256=hashes, figures=figures, heatmap_mae_max=max_mae,
                  note='Repeated calls are not independent episodes. Timepoint changes state/action context as well as reward. Reward is removed only from target history; few-shot text is fixed. No confidence intervals or significance tests.')
    (OUT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    print(f'Output: {OUT}')


if __name__ == '__main__':
    main()
