"""Offline reward/model bars from saved Exp.11 responses; never sends requests."""
import sys
import csv
import json
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments import common
from experiments.common_analysis.joint_aggregate import accuracy_rows, continuous_rows
from experiments.common_analysis.joint_data import parse_response
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ORDER = [(reward, model) for reward in (True, False) for model in ('terra', 'luna')]
METRICS = (
    ('accuracy', '正解率', 'Accuracy (%)', '高いほど良い'),
    ('bin_mae', 'binの誤差 MAE', 'Mean absolute bin error', '低いほど良い'),
    ('nrmse', 'action値の NRMSE', 'NRMSE (RMSE / 4)', '低いほど良い'),
    ('pearson', 'action値の相関係数', 'Pearson r', '高いほど良い'),
)


def aggregate(root, *, group_field='reward_present', order=ORDER):
    manifest = common.read_json(root / 'api/manifest.json')
    records = common.read_jsonl(root / 'api/records.jsonl')
    queries = {q['query_id']: q for q in manifest['queries']}
    if (len(queries) != 40 or len(records) != 40
            or {r['query_id'] for r in records} != set(queries)):
        raise ValueError('Expected all 40 unique saved responses')
    if Counter((r[group_field], r['model_alias']) for r in records) != Counter(dict.fromkeys(order, 10)):
        raise ValueError('Expected four conditions with N10')
    for r in records:
        if any(r.get(k) != v for k, v in queries[r['query_id']].items()):
            raise ValueError('Record differs from manifest')
    table = []
    for group, model in order:
        selected = sorted([r for r in records if (r[group_field], r['model_alias']) == (group, model)],
                          key=lambda r: r['ordinal'])
        if [r['ordinal'] for r in selected] != list(range(10)):
            raise ValueError('Missing target ordinal')
        parsed = [{**r, 'components': parse_response(r['assistant_text'], r['task'], r['metric']),
                   'gt_action_bin': r['ground_truth'], 'gt_action_value': r['action_value_ground_truth']}
                  for r in selected]
        bins, _, _ = accuracy_rows(parsed)
        continuous = continuous_rows(parsed, 'action_value', [4.0], ('torque',), 1000, 0)
        valid_bins = [r for r in parsed if r['components']['action_bin']['ok']]
        bin_errors = np.asarray([r['components']['action_bin']['value'][0]
                                 - r['gt_action_bin'][0] for r in valid_bins])
        samples = {'accuracy': 100.0 * (bin_errors == 0), 'bin_mae': np.abs(bin_errors)}
        for row in bins + continuous:
            if row['metric_name'] in {m[0] for m in METRICS}:
                for key in list(row):
                    if key.startswith('bootstrap_'):
                        del row[key]
                values = samples.get(row['metric_name'])
                row.update(std=float(np.std(values, ddof=1)) if values is not None and len(values) > 1 else None,
                           std_kind='query_sample_sd_ddof1' if values is not None else 'not_defined_per_query')
                row[group_field] = group
                if group_field == 'reward_present':
                    row['reward_mode'] = 'with_reward' if group else 'without_reward'
                table.append(row)
    return table


BASELINE_SOURCE = common.EXPERIMENTS / '07_view_rawdata/reasoning_score_analysis/pendulum_score_performance/joined_samples.csv'


def load_baselines(root):
    with BASELINE_SOURCE.open(encoding='utf-8', newline='') as handle:
        source = list(csv.DictReader(handle))
    targets = common.read_json(root / 'api/manifest.json')['queries']
    baselines = []
    for model in ('terra', 'luna'):
        selected = sorted([r for r in source if r['model'] == model and r['Task'] == 'Pendulum-v1'
                           and r['Metrics'] == 'next-action' and r['H'] == '20'], key=lambda r: int(r['ordinal']))
        if [int(r['ordinal']) for r in selected] != list(range(10)):
            raise ValueError('Baseline must contain the same ten targets')
        for r in selected:
            paired = [q for q in targets if q['model_alias'] == model and q['ordinal'] == int(r['ordinal'])]
            if not paired or any(
                    r['episode_path'] != q['episode_path'] or int(r['query_index']) != q['query_index']
                    or json.loads(r['gt_action_value']) != q['action_value_ground_truth'] for q in paired):
                raise ValueError('Baseline target identity differs')
        bins = [r for r in selected if r['bin_valid'] == 'True']
        actions = [r for r in selected if r['torque_valid'] == 'True']
        errors = np.asarray([float(r['pred_bin']) - float(r['true_bin']) for r in bins])
        action_errors = np.asarray([float(r['pred_torque']) - float(r['true_torque']) for r in actions])
        baselines.append(dict(model_alias=model, n=10, parsed_bins=len(bins), parsed_actions=len(actions),
                              accuracy=float(100 * np.mean(errors == 0)), bin_mae=float(np.mean(np.abs(errors))),
                              nrmse=float(np.sqrt(np.mean(action_errors ** 2)) / 4)))
    return baselines


def draw(ax, table, metric, baselines, *, group_field='reward_present', order=ORDER,
         group_labels=('rewardあり', 'rewardなし')):
    name, title, ylabel, direction = metric
    rows = [next(r for r in table if r['metric_name'] == name
                 and (r[group_field], r['model_alias']) == pair) for pair in order]
    positions = [0, 1, 3, 4]
    values = [r['value'] for r in rows]
    ax.bar(positions, [v if v is not None else 0 for v in values], width=0.68,
           color=['#3568A8', '#D78937'] * 2, zorder=3)
    errors = [r['std'] if r['std'] is not None else 0 for r in rows]
    if any(r['std'] is not None for r in rows):
        ax.errorbar(positions, [v if v is not None else np.nan for v in values], yerr=errors,
                    fmt='none', ecolor='#263445', elinewidth=1.5, capsize=5, capthick=1.5, zorder=4)
    if name != 'pearson':
        reference = {r['model_alias']: r[name] for r in baselines}
        for x, (_, model) in zip(positions, order):
            ax.hlines(reference[model], x - 0.34, x + 0.34, color='#D62828', linewidth=3,
                      zorder=6, label='通常Prompt' if x == 0 else None)
        ax.legend(loc='upper right' if name == 'accuracy' else 'upper left', frameon=False, fontsize=9)
    ax.set_xticks(positions, ['Terra', 'Luna', 'Terra', 'Luna'])
    ax.set_xlim(-0.8, 4.8)
    ax.axvline(2, color='#D1D5DB', linewidth=1)
    ax.axhline(0, color='#9CA3AF', linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{title}  |  {direction}', loc='left', fontsize=13, pad=14)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#E5E7EB', linewidth=0.7)
    ax.spines[['top', 'right']].set_visible(False)
    if name == 'accuracy':
        ax.set_ylim(0, 110)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
    elif name == 'pearson':
        ax.set_ylim(-1.16, 1.16)
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    else:
        ax.set_ylim(min(0, min(v-e for v,e in zip(values, errors)) - 0.05 * max(values)), max([v + e for v, e in zip(values, errors) if v is not None] + [r[name] for r in baselines] + [0.01]) * 1.3)
    if name == 'bin_mae':
        ax.set_ylim(bottom=0)
    for x, value, error, row in zip(positions, values, errors, rows):
        label = 'N/A' if value is None else (f'{value:.0f}%' if name == 'accuracy' else f'{value:.3f}')
        label += f"\nn={row['valid_n']}"
        negative = value is not None and value < 0
        ax.annotate(label, (x, (value or 0) + (-error if negative else error)), xytext=(0, -7 if negative else 7),
                    textcoords='offset points', ha='center', va='top' if negative else 'bottom', fontsize=10)
    for center, label in zip((0.5, 3.5), group_labels):
        ax.text(center, -0.14, label, transform=ax.get_xaxis_transform(), ha='center', fontsize=12)


def main(*, root=None, group_field='reward_present', order=ORDER,
         group_labels=('rewardあり', 'rewardなし'), title='ICR-FQI K=5',
         overview_name='reward_comparison'):
    root = HERE / 'results' if root is None else root
    table = aggregate(root, group_field=group_field, order=order)
    baselines = load_baselines(root)
    output = common.output_path(root / 'analysis/plots')
    output.mkdir(parents=True, exist_ok=True)
    common.write_csv(root / 'analysis/bar_metrics.csv', table)
    common.write_csv(root / 'analysis/bar_baselines.csv', baselines)
    common.write_json(root / 'analysis/bar_metrics.json', {
        'source': 'api/records.jsonl', 'source_sha256': common.file_sha256(root / 'api/records.jsonl'),
        'definitions': {'accuracy': 'Exact final action-bin match, percent of parsed bins',
                        'bin_mae': 'Mean absolute difference of final predicted and true bin IDs',
                        'nrmse': 'Continuous action RMSE / theoretical action range width 4 ([-2, 2])',
                        'pearson': 'Pearson correlation between predicted and true continuous actions'},
        'baseline_source': str(BASELINE_SOURCE.relative_to(common.ROOT)),
        'baseline_source_sha256': common.file_sha256(BASELINE_SOURCE), 'baselines': baselines,
        'note': 'Each metric uses its own parsed component. Accuracy and bin MAE error bars are +/-1 '
                'sample SD of the ten query outcomes (ddof=1); no SE or bootstrap. NRMSE and Pearson '
                'have no per-query sample SD and are drawn without error bars. Red segments show '
                'the same-model standard-prompt, rewards-present baseline from 05 reused by 06/07. '
                'The ten target windows overlap within episode_0.',
        'metrics': table})
    plt.rcParams.update({'font.family': ['Hiragino Sans', 'DejaVu Sans'],
                         'font.size': 11, 'axes.unicode_minus': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, metric in zip(axes.flat, METRICS):
        draw(ax, table, metric, baselines, group_field=group_field, order=order, group_labels=group_labels)
    fig.suptitle(f'{title}  |  Pendulum Next Action  |  H=20, N=10 / condition', fontsize=16, y=0.98)
    fig.subplots_adjust(left=0.085, right=0.98, top=0.90, bottom=0.12, hspace=0.52, wspace=0.28)
    fig.text(0.5, 0.025, 'NRMSE = RMSE / 4    •    相関係数 = Pearson r    •    エラーバー = N10の標本STD（正解率・bin MAEのみ）', ha='center', fontsize=10, color='#4B5563')
    for suffix in ('png', 'svg'):
        fig.savefig(output / f'{overview_name}.{suffix}', dpi=180, facecolor='white')
    plt.close(fig)
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(7, 5))
        draw(ax, table, metric, baselines, group_field=group_field, order=order, group_labels=group_labels)
        fig.subplots_adjust(left=0.14, right=0.97, top=0.87, bottom=0.23)
        fig.text(0.5, 0.025, f'{title} | H20 | N10 / condition | Error bars: N10 sample STD (accuracy / bin MAE only)', ha='center', fontsize=9)
        for suffix in ('png', 'svg'):
            fig.savefig(output / f'{metric[0]}.{suffix}', dpi=180, facecolor='white')
        plt.close(fig)
    print(f'Saved 4 metric charts + overview (PNG/SVG): {output}')
    for group, model in order:
        label = ('with_reward' if group else 'without_reward') if group_field == 'reward_present' else group
        print(model, label,
              {r['metric_name']: r['value'] for r in table
               if (r[group_field], r['model_alias']) == (group, model)})


if __name__ == '__main__':
    main()
