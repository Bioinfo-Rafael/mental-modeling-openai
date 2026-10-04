"""Seven three-panel comparisons, using the saved Joint metrics and manual scores."""
import argparse
from collections import Counter
from importlib import import_module
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
data = import_module('experiments.10_ICRL.analysis.results_data')
# Reuse Exp.09's metric definitions, aggregation, annotations and axis styling.
previous = import_module('experiments.09_ablation_reward.analysis.plot_comparisons')
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np

HERE = data.HERE
MODELS, METHODS, EPISODES = data.MODELS, data.METHODS, data.EPISODES
METRICS, COLORS = previous.METRICS, previous.COLORS
aggregate, find, require = previous.aggregate, previous.find, previous.require


def make_tables(samples):
    prompt = aggregate(samples, ('method',), 18)
    prompt_models = aggregate(samples, ('model', 'method'), 9)
    episodes = aggregate(samples, ('E',), 24)
    episode_models = aggregate(samples, ('model', 'E'), 12)
    matrix = aggregate(samples, ('method', 'E'), 6)
    matrix_models = aggregate(samples, ('model', 'method', 'E'), 3)
    tables = dict(joined_samples=samples,
                  prompt_summary=[dict(model='pooled', **r) for r in prompt] + prompt_models,
                  episode_summary=[dict(model='pooled', **r) for r in episodes] + episode_models,
                  matrix_summary=[dict(model='pooled', **r) for r in matrix] + matrix_models)
    for rows, count in ((prompt, 4), (prompt_models, 8), (episodes, 3),
                        (episode_models, 6), (matrix, 12), (matrix_models, 24)):
        require(len(rows) == count and sum(r['n'] for r in rows) == 72, 'Unexpected aggregation size')
        require(sum(r['matches'] for r in rows) == sum(r['correct'] for r in samples), 'Incorrect match totals')
    return tables


def save(fig, output, name):
    path = output / 'figures' / name
    fig.savefig(path, dpi=180, facecolor='white')
    plt.close(fig)
    return str(path.relative_to(output))


def bars(rows, axis, values, labels, *, by_model, max_mae, output, name):
    scopes = MODELS if by_model else ('pooled',)
    n = find(rows, model=scopes[0], **{axis: values[0]})['n']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.subplots_adjust(left=.06, right=.98, bottom=.14, top=.68, wspace=.30)
    varying = 'Prompt' if axis == 'method' else 'Context episode count E'
    pooling = 'E and repeats' if axis == 'method' else 'prompt conditions and repeats'
    fig.suptitle(f'{varying} comparison · {"by model" if by_model else "both models pooled"}', fontsize=20, y=.98)
    fig.text(.5, .89, f'Pooled over {pooling}{"" if by_model else " and models"}; n={n} per bar.',
             ha='center', fontsize=11, color='#475569')
    colors = COLORS if by_model else ('#527B9E',)
    if by_model:
        fig.legend([Patch(color=c) for c in colors], ['Terra', 'Luna'], loc='upper center',
                   bbox_to_anchor=(.5, .855), ncol=2, frameon=False)
    for ax, metric in zip(axes, METRICS):
        previous.bar_axis(ax, metric, max_mae)
        for i, scope in enumerate(scopes):
            group = [find(rows, model=scope, **{axis: v}) for v in values]
            positions = np.arange(len(values)) + ((i-.5)*.36 if by_model else 0)
            drawn = ax.bar(positions, [r[metric[0]] for r in group],
                           width=.33 if by_model else .62, color=colors[i])
            previous.annotate(ax, drawn, group, metric)
        if by_model:
            # Keep adjacent 100.0% labels separated when both models are perfect.
            for text in ax.texts:
                text.set_fontsize(9)
        ax.set_xticks(range(len(values)), labels)
        ax.set_xlim(-.6, len(values)-.4)
    return save(fig, output, name)


def heatmaps(rows, scope, max_mae, output, name):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    fig.subplots_adjust(left=.10, right=.96, bottom=.12, top=.72, wspace=.50)
    title = 'Both models pooled' if scope == 'pooled' else scope.title()
    n = 6 if scope == 'pooled' else 3
    fig.suptitle(f'{title} · Prompt × context episode count E', fontsize=20, y=.98)
    fig.text(.5, .90, f'n={n} per cell. Means across repeated calls; accuracy is a percentage.', ha='center')
    for ax, metric in zip(axes, METRICS):
        key, title, label, fmt, cmap = metric
        matrix = np.array([[find(rows, model=scope, method=method, E=E)[key]
                            for E in EPISODES] for method in METHODS])
        lo, hi = {'accuracy_pct': (0, 100), 'action_mae': (0, max_mae), 'reasoning_mean': (1, 4)}[key]
        im = ax.imshow(matrix, vmin=lo, vmax=hi, cmap=cmap, aspect='auto')
        ax.set_title(title, fontsize=14, pad=18)
        ax.set_xticks(range(3), [f'E={E}' for E in EPISODES])
        ax.set_yticks(range(4), data.METHOD_LABELS)
        ax.tick_params(length=0, pad=8)
        for (i, j), value in np.ndenumerate(matrix):
            text = format(value, fmt) + ('%' if key == 'accuracy_pct' else '')
            ax.text(j, i, f'{text}\nn={n}', ha='center', va='center', fontsize=11,
                    color='white' if (value-lo)/(hi-lo) > .55 else '#17212B')
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04, shrink=.88, label=label)
    return save(fig, output, name)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Generate 7 comparison figures after all 72 reasoning scores are complete.')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'results/analysis/comparisons',
                        help='New output directory; existing directories are never overwritten')
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError('Output directory already exists; choose a new --output-dir')
    # Validate all inputs before creating a directory, CSV or figure.
    samples, hashes = data.load_samples()
    tables = make_tables(samples)
    cells = [r for r in tables['matrix_summary'] if r['model'] in MODELS]
    max_mae = math.ceil(max(r['action_mae'] for r in cells)*10)/10 or .1
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    (output / 'figures').mkdir()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    for name, rows in tables.items():
        previous.write_csv(output / f'{name}.csv', rows)
    figures = []
    for axis, values, labels, table, first in (
        ('method', METHODS, ('Direct', 'PPO', 'ICRL\n(K=2)', 'ICRL\n(K=5)'), 'prompt_summary', 1),
        ('E', EPISODES, ('E=1', 'E=3', 'E=9'), 'episode_summary', 3),
    ):
        for offset, by_model in enumerate((False, True)):
            scope = 'by_model' if by_model else 'pooled'
            name = f'{first+offset:02d}_{"prompt" if axis == "method" else "episodes"}_{scope}.png'
            figures.append(bars(tables[table], axis, values, labels, by_model=by_model,
                                max_mae=max_mae, output=output, name=name))
    for number, scope in enumerate(('pooled', *MODELS), 5):
        figures.append(heatmaps(tables['matrix_summary'], scope, max_mae, output,
                                f'{number:02d}_matrix_{scope}.png'))
    report = dict(samples=len(samples), source_sha256=hashes, figures=figures,
                  matches=sum(r['correct'] for r in samples),
                  reasoning_counts=dict(Counter(r['reasoning_score'] for r in samples)),
                  n_per_group=dict(prompt_pooled=18, prompt_by_model=9, episodes_pooled=24,
                                   episodes_by_model=12, matrix_pooled=6, matrix_by_model=3),
                  heatmap_mae_max=max_mae, excluded_samples=0,
                  note='Equal-weight sample pooling. Repeats share one target query, not independent episodes. Reasoning is the mean ordinal score (1–4, higher is stronger), not correctness or verified internal FQI execution. No confidence intervals or significance tests.')
    (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'{len(figures)} figures / {len(samples)} responses: {output}')


if __name__ == '__main__':
    main()
