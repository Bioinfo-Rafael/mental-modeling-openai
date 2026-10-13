"""Offline, paired LAG comparison figures from verified saved responses."""
import argparse
from collections import defaultdict
from datetime import datetime
import html
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_results_viewer import load_data, HERE

MODELS = ('sol', 'terra', 'luna')
COLORS = {'sol': '#B84B96', 'terra': '#2875B5', 'luna': '#DF871E'}
SCOPES = ('self', 'all_aircraft_missiles')
HISTORIES = (2, 5, 10, 20, 30)
FIELDS = ('lon_deg', 'lat_deg', 'alt_m', 'roll_deg', 'pitch_deg', 'yaw_deg',
          'tas_mps', 'vn_mps', 've_mps', 'vu_mps')
ACTION_FIELDS = ('altitude_action', 'heading_action', 'speed_action')
UNITS = ('deg', 'deg', 'm', 'deg', 'deg', 'deg', 'm/s', 'm/s', 'm/s', 'm/s')
CIRCULAR = (3, 5)
OVERALL_METRICS = ('state_joint_nrmse', 'state_joint_zrmse', 'state_joint_nmae', 'state_joint_zmae')
OVERALL_LABELS = {
    'state_joint_nrmse': '10D overall RMSE | range-normalized',
    'state_joint_zrmse': '10D overall RMSE | SD-normalized (z-score scale)',
    'state_joint_nmae': '10D overall MAE | range-normalized',
    'state_joint_zmae': '10D overall MAE | SD-normalized (z-score scale)',
}
OVERALL_FORMULAS = {
    'state_joint_nrmse': 'sqrt(mean over queries and 10 dimensions of (residual / reference range)**2)',
    'state_joint_zrmse': 'sqrt(mean over queries and 10 dimensions of (residual / reference SD)**2)',
    'state_joint_nmae': 'mean over queries and 10 dimensions of abs(residual / reference range)',
    'state_joint_zmae': 'mean over queries and 10 dimensions of abs(residual / reference SD)',
}
NEIGHBOR_METRICS = {
    'action_neighbor_mae': ('Action MAE from adjacent recorded action', 'Mean absolute ID difference (bins)', 'Action adjacent MAE'),
    'state_change_ratio_range': ('State change magnitude ratio | range-normalized', 'Predicted change / actual change', 'State change ratio (range)'),
    'state_change_ratio_sd': ('State change magnitude ratio | SD-normalized', 'Predicted change / actual change', 'State change ratio (SD)'),
}


def figure_directory(cohort, category, name):
    if category == 'Action': group = 'action/overall'
    elif category == 'Action components': group = 'action/components'
    elif category == 'Confusion matrices': group = 'action/confusion'
    elif category == 'State overall (10D)': group = 'state/overall'
    elif category == 'State overview': group = 'state/component_rmse_mean'
    elif category == 'Tokens': group = 'tokens'
    elif category == 'Action adjacent MAE': group = 'action/adjacent_mae'
    elif category == 'State change ratio (range)': group = 'state/change_ratio/range'
    elif category == 'State change ratio (SD)': group = 'state/change_ratio/sd'
    elif category == 'State components':
        metric = next(k for k in ('mae', 'nrmse', 'zrmse') if name.startswith(f'state_state_{k}_'))
        group = f'state/components/{metric}'
    else: raise ValueError(f'Unknown figure category: {category}')
    return Path(group) / cohort


def state_errors(prediction, truth):
    errors = np.asarray(prediction, float) - np.asarray(truth, float)
    errors[..., list(CIRCULAR)] = (errors[..., list(CIRCULAR)] + 180) % 360 - 180
    return errors


def reference_scales(samples):
    """One fixed reference across cohorts/H/tasks/models, unique target states."""
    unique = {}
    for s in samples.values():
        if not s['task'].endswith('state'):
            continue
        key = (s['episode_id'], s['target_id'], s['target_step'])
        if key in unique and unique[key] != s['ground_truth']:
            raise ValueError('Conflicting truth for a target state')
        unique[key] = s['ground_truth']
    values = np.array(list(unique.values()), float)
    reference = values.copy()
    origins = {}
    # A fixed cut at the largest empty angular gap prevents 359/0 crossings
    # from inflating range or SD. Apply the same reference to every figure.
    for i in CIRCULAR:
        angles = np.unique(values[:, i] % 360)
        gaps = np.diff(np.r_[angles, angles[0] + 360])
        origin = angles[(int(np.argmax(gaps)) + 1) % len(angles)]
        reference[:, i] = (values[:, i] - origin) % 360 + origin
        origins[FIELDS[i]] = float(origin)
    return dict(n_unique_target_states=len(unique), fields=list(FIELDS),
                minimum=reference.min(axis=0).tolist(), maximum=reference.max(axis=0).tolist(),
                mean=reference.mean(axis=0).tolist(), range=np.ptp(reference, axis=0).tolist(),
                sd=reference.std(axis=0, ddof=0).tolist(), circular_origins_deg=origins)


def prepare(data):
    grouped = defaultdict(list)
    for row in data['rows']:
        row = dict(row)
        s = data['samples'][row['sample_id']]
        row['truth'] = s['ground_truth']
        row['prediction'] = (row['scoring'] or {}).get('prediction')
        row['usage'] = (row['raw_response'] or {}).get('usage') or {}
        grouped[(row['episode_id'], row['model_alias'], row['H'], row['task'], row['observation_scope'])].append(row)
    for key, group in grouped.items():
        if len(group) != 10 or len({(r['target_id'], r['sample_number']) for r in group}) != 10:
            raise ValueError(f'Expected two targets x five windows: {key}')
    return grouped


def usable(row, metric):
    if metric in NEIGHBOR_METRICS:
        return row.get('neighbor_metrics', {}).get(metric) is not None
    if metric.startswith('token_'):
        key = metric.removeprefix('token_')
        if key == 'reasoning':
            return row['usage'].get('completion_tokens_details', {}).get('reasoning_tokens') is not None
        return row['usage'].get(key) is not None
    return row['prediction'] is not None


def estimate(rows, metric, dimension, scales):
    if not rows:
        return None
    if metric in NEIGHBOR_METRICS:
        return float(np.mean([r['neighbor_metrics'][metric] for r in rows]))
    if metric.startswith('token_'):
        key = metric.removeprefix('token_')
        return float(np.mean([r['usage']['completion_tokens_details']['reasoning_tokens']
                              if key == 'reasoning' else r['usage'][key] for r in rows]))
    p = np.array([r['prediction'] for r in rows], float)
    g = np.array([r['truth'] for r in rows], float)
    if metric == 'action_accuracy':
        values = np.all(p == g, axis=1) if dimension is None else p[:, dimension] == g[:, dimension]
        return float(100 * np.mean(values))
    if metric == 'action_bin_mae':
        return float(np.mean(np.abs(p-g)))
    errors = state_errors(p, g)
    if metric in OVERALL_METRICS:
        denominator = np.array(scales['sd' if metric in ('state_joint_zrmse', 'state_joint_zmae') else 'range'])
        if np.any(denominator <= 0): return None
        standardized = errors / denominator
        return float(np.mean(np.abs(standardized)) if metric.endswith('mae')
                     else np.sqrt(np.mean(standardized**2)))
    if metric == 'state_mae':
        values = np.mean(np.abs(errors), axis=0)
    else:
        denominator = np.array(scales['range' if metric == 'state_nrmse' else 'sd'])
        values = np.divide(np.sqrt(np.mean(errors**2, axis=0)), denominator,
                           out=np.full(10, np.nan), where=denominator > 0)
    value = values[dimension] if dimension is not None else (np.mean(values) if np.isfinite(values).all() else np.nan)
    return float(value) if np.isfinite(value) else None


def paired_estimates(left, right, metric, dimension, scales):
    a = {r['comparison_id']: r for r in left}
    b = {r['comparison_id']: r for r in right}
    if set(a) != set(b) or len(a) != len(left) or len(b) != len(right):
        raise ValueError('Input scopes must have identical unique target windows')
    ids = [key for key in a if usable(a[key], metric) and usable(b[key], metric)]
    for key in ids:
        if a[key]['truth'] != b[key]['truth']:
            raise ValueError('Paired ground truth differs')
    x = estimate([a[key] for key in ids], metric, dimension, scales)
    y = estimate([b[key] for key in ids], metric, dimension, scales)
    return (y-x if x is not None and y is not None else None), len(ids), x, y


class Plotter:
    def __init__(self, data, output):
        self.data, self.output = data, output
        self.grouped = prepare(data)
        self.scales = reference_scales(data['samples'])
        self.episodes = sorted({r['episode_id'] for r in data['rows']})
        self.inventory, self.metrics = [], []

    def selected(self, cohort, model, h, task, scope):
        episodes = self.episodes if cohort == 'pooled' else [cohort]
        return [r for ep in episodes for r in self.grouped[(ep, model, h, task, scope)]]

    def save(self, fig, cohort, category, name, title):
        relative = figure_directory(cohort, category, name)
        folder = self.output / relative
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / f'{name}.png', dpi=130, facecolor='white')
        plt.close(fig)
        self.inventory.append(dict(cohort=cohort, category=category, name=name, title=title,
                                   png=str(relative / f'{name}.png')))

    def lines(self, cohort, kind, metric, dimension=None):
        expected = 30 if cohort == 'pooled' else 10
        if metric in NEIGHBOR_METRICS:
            title, unit, category = NEIGHBOR_METRICS[metric]
            diffunit = 'Difference (all minus self)'
        elif metric == 'action_accuracy':
            title = 'Action accuracy' + (' (all 3 IDs match)' if dimension is None else f' | {ACTION_FIELDS[dimension]}')
            unit, diffunit = 'Accuracy (%)', 'Difference (percentage points)'
            category = 'Action' if dimension is None else 'Action components'
        elif metric == 'action_bin_mae':
            title, unit, diffunit, category = 'Action bin MAE (mean over 3 IDs)', 'Mean absolute ID error (bins)', 'Difference (bins)', 'Action'
        elif metric in OVERALL_METRICS:
            title, unit, diffunit, category = OVERALL_LABELS[metric], 'Dimensionless error', 'Difference (dimensionless)', 'State overall (10D)'
        elif metric.startswith('state_'):
            label = {'state_mae': 'MAE', 'state_nrmse': 'NRMSE = RMSE / reference range',
                     'state_zrmse': 'Z-scaled RMSE = RMSE / reference SD'}[metric]
            title = f'{label} | ' + ('macro mean of 10 dimensions' if dimension is None else FIELDS[dimension])
            unit = UNITS[dimension] if metric == 'state_mae' else 'Dimensionless error'
            diffunit, category = f'Difference ({unit})', 'State overview' if dimension is None else 'State components'
        else:
            title = f'{kind.title()} tasks | Mean ' + metric.removeprefix('token_').replace('_', ' ')
            unit, diffunit, category = 'Tokens per request', 'Difference (tokens)', 'Tokens'
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
        fig.subplots_adjust(left=.07, right=.98, bottom=.17, top=.79, hspace=.42, wspace=.30)
        cohort_label = 'All 3 episodes pooled' if cohort == 'pooled' else cohort
        fig.suptitle(title, fontsize=17, y=.975)
        fig.text(.5, .925, f'{cohort_label} | 2 aircraft | planned n={expected} per point', ha='center', fontsize=11)
        handles = []
        for task_i, prefix in enumerate(('next', 'last')):
            task = f'{prefix}-{kind}'
            for model_i, model in enumerate(MODELS):
                series = [[], [], []]
                for h in HISTORIES:
                    groups = [self.selected(cohort, model, h, task, scope) for scope in SCOPES]
                    for col, group in enumerate(groups):
                        valid = [r for r in group if usable(r, metric)]
                        val = estimate(valid, metric, dimension, self.scales)
                        series[col].append((val, len(valid)))
                        self.metrics.append(dict(cohort=cohort, task=task, model=model, H=h,
                                                 scope=SCOPES[col], metric=metric, dimension=dimension,
                                                 value=val, n_valid=len(valid), n_planned=expected))
                    delta, n, a, b = paired_estimates(*groups, metric, dimension, self.scales)
                    series[2].append((delta, n))
                    self.metrics.append(dict(cohort=cohort, task=task, model=model, H=h,
                                             scope='all_minus_self', metric=metric, dimension=dimension,
                                             value=delta, n_valid=n, n_planned=expected,
                                             paired_self=a, paired_all=b))
                for col, points in enumerate(series):
                    ax = axes[task_i, col]
                    line, = ax.plot(HISTORIES, [v if v is not None else np.nan for v, n in points],
                                    color=COLORS[model], marker=('o', 's', '^')[model_i],
                                    markersize=5, linewidth=1.7, label=model.title())
                    if task_i == 0 and col == 0:
                        handles.append(line)
                    for h, (v, n) in zip(HISTORIES, points):
                        if n != expected and v is not None:
                            ax.annotate(f'n={n}', (h, v), xytext=(2, 9+11*model_i),
                                        textcoords='offset points', fontsize=8, color=COLORS[model])
                    if all(v is None for v, n in points):
                        ax.text(.5, .5, 'Undefined: zero reference scale or no data', transform=ax.transAxes,
                                ha='center', fontsize=8)
            for col, scope in enumerate(('Self', 'All aircraft + missiles', 'All minus self (paired)')):
                ax = axes[task_i, col]
                ax.set_title(f'{task} | {scope}', fontsize=11)
                ax.set_xticks(HISTORIES)
                ax.set_xlabel('History length H')
                ax.set_ylabel(diffunit if col == 2 else unit)
                ax.grid(alpha=.22)
                ax.spines[['top', 'right']].set_visible(False)
                if col == 2:
                    ax.axhline(0, color='#555', linewidth=.9)
                elif metric.startswith('state_change_ratio_'):
                    ax.axhline(1, color='#777', linewidth=.9, linestyle='--')
                    ax.set_ylim(bottom=0)
                elif metric == 'action_accuracy':
                    ax.set_ylim(-3, 103)
                else:
                    ax.set_ylim(bottom=0)
            # Equal self/all scale within task. Difference keeps its own scale.
            if metric != 'action_accuracy':
                top = max(axes[task_i, c].get_ylim()[1] for c in (0, 1))
                for c in (0, 1): axes[task_i, c].set_ylim(0, top)
        fig.legend(handles, [m.title() for m in MODELS], loc='upper center', bbox_to_anchor=(.5,.895), ncol=3, frameon=False)
        improvement = 'positive = improved accuracy' if metric == 'action_accuracy' else ('positive = more tokens' if metric.startswith('token_') else 'negative = reduced error')
        note = f'Each point pools distinct windows; H can change target times. Difference: {improvement}.\n'
        note += 'Missing/invalid values excluded; labels mark reduced n. Difference uses only valid paired self/all responses.'
        if metric.startswith('state_'):
            note += '\nRoll/yaw residuals use shortest angles. Normalization reference is fixed across all cohorts, models, scopes, H and tasks.'
        if metric == 'token_reasoning':
            note += '\nReasoning tokens are included in completion tokens; missing usage is not treated as zero.'
        if metric in NEIGHBOR_METRICS:
            note = 'Next: compare prediction at t to actual t-1. Last: compare prediction at t to actual t+1.\n'
            if metric == 'action_neighbor_mae':
                note += '0 = same action IDs as neighbor; this is not ground-truth prediction error. Last-action t+1 actions were not supplied.\n'
            else:
                note += 'Per-query L2 norm ratio, then mean. 0 = neighbor copy; 1 = same change magnitude (not necessarily direction).\n'
                note += 'Zero actual change is undefined and excluded; angles wrapped; fixed per-dimension scales. '
            note += 'Reduced n labeled. Difference uses common valid pairs; positive/negative is not inherently better.'
        fig.text(.5, .035, note, ha='center', va='bottom', fontsize=8.5, color='#4B5563', linespacing=1.6)
        suffix = '' if dimension is None else '_' + (ACTION_FIELDS if kind == 'action' else FIELDS)[dimension]
        self.save(fig, cohort, category, f'{kind}_{metric}{suffix}', title)

    def confusion(self, cohort, task, dimension):
        count = (3, 5, 3)[dimension]
        labels = (('up', 'hold', 'down'), ('left30', 'left15', 'hold', 'right15', 'right30'), ('up', 'hold', 'down'))[dimension]
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        fig.subplots_adjust(left=.09, right=.91, bottom=.15, top=.85, wspace=.42, hspace=.48)
        field = ACTION_FIELDS[dimension]
        title = f'{task} | {field} confusion matrices'
        expected = (30 if cohort == 'pooled' else 10)*len(HISTORIES)
        fig.suptitle(title, y=.98, fontsize=17)
        fig.text(.5, .935, f'{cohort} | H pooled | planned n={expected} per matrix', ha='center')
        for row_i, scope in enumerate(SCOPES):
            for col, model in enumerate(MODELS):
                rows = [r for h in HISTORIES for r in self.selected(cohort, model, h, task, scope)]
                valid = [r for r in rows if r['prediction'] is not None]
                matrix = np.zeros((count, count), int)
                for r in valid: matrix[r['truth'][dimension], r['prediction'][dimension]] += 1
                denom = matrix.sum(axis=1, keepdims=True)
                pct = np.divide(matrix*100., denom, out=np.full(matrix.shape, np.nan), where=denom>0)
                ax = axes[row_i, col]
                im = ax.imshow(np.ma.masked_invalid(pct), vmin=0, vmax=100, cmap='Blues')
                for i in range(count):
                    for j in range(count):
                        text = f'{matrix[i,j]}\n{pct[i,j]:.0f}%' if denom[i,0] else '0\nN/A'
                        ax.text(j, i, text, ha='center', va='center', fontsize=9,
                                color='white' if pct[i,j]>55 else '#202C3B')
                ax.set_xticks(range(count), [f'{i}: {v}' for i,v in enumerate(labels)], rotation=25, ha='right')
                ax.set_yticks(range(count), [f'{i}: {v}' for i,v in enumerate(labels)])
                ax.set_xlabel('Predicted ID'); ax.set_ylabel('True ID')
                ax.set_title(f'{model.title()} | {"self" if row_i==0 else "all"} | n={len(valid)}', fontsize=11)
                self.metrics.append(dict(cohort=cohort, task=task, model=model, scope=scope,
                                         metric='confusion', dimension=dimension, H='pooled',
                                         counts=matrix.tolist(), n_valid=len(valid), n_planned=expected))
        cax = fig.add_axes([.94,.27,.013,.46])
        fig.colorbar(im, cax=cax, label='Within true class (%)')
        fig.text(.5,.025,'Cells: count and row percentage. Classes without true examples are N/A.\nH-pooled windows may overlap; counts are not independent episodes.',ha='center',fontsize=9)
        self.save(fig, cohort, 'Confusion matrices', f'confusion_{task}_{field}', title)


def write_gallery(output, inventory):
    cards = ''.join(f'<article data-cohort="{html.escape(r["cohort"])}" data-category="{html.escape(r["category"])}"><h2>{html.escape(r["title"])}</h2><p>{html.escape(r["cohort"])}</p><a href="{r["png"]}"><img loading="lazy" src="{r["png"]}" alt="{html.escape(r["title"])}"></a><p><a href="{r["png"]}">PNG</a></p></article>' for r in inventory)
    labels = {'pooled':'全試合pool', 'Action':'行動：全体', 'Action components':'行動：成分別',
              'Confusion matrices':'行動：混同行列', 'State overall (10D)':'状態：10成分全体',
              'State overview':'状態：成分別RMSEの平均', 'State components':'状態：成分別', 'Tokens':'token数',
              'Action adjacent MAE':'行動：隣接実測値との差MAE', 'State change ratio (range)':'状態：変化量比（範囲基準）',
              'State change ratio (SD)':'状態：変化量比（標準偏差基準）'}
    def options(values):
        return ''.join(f'<option value="{html.escape(v)}" {"selected" if v == "State overall (10D)" else ""}>{html.escape(labels.get(v,v))}</option>' for v in dict.fromkeys(values))
    page = '''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LAG comparison figures</title><style>body{max-width:1500px;margin:auto;padding:24px;font:15px/1.7 system-ui;background:#f4f7fb;color:#172333}article{background:white;border:1px solid #d5ddeb;border-radius:12px;padding:20px;margin:20px 0}img{width:100%;height:auto}h2{font-size:19px}select{padding:8px;font:inherit;margin:8px}a{color:#2457cf}[hidden]{display:none}nav{position:sticky;top:0;background:#f4f7fb;padding:10px;border-bottom:1px solid #d5ddeb}</style><h1>実験14 · LAG 比較図</h1><p>全試合pool: 各点n=30（5問×2機体×3試合）。試合別: 各点n=10。状態の解析失敗は除外し、有効nが減った点に注記。</p><p>折れ線図: 行＝next／last、列＝self／全機体／全機体−self。差分は同じ問題の有効なペアのみ。正解率は正の差、誤差は負の差が改善。</p><p>状態はMAE、NRMSE（RMSE÷正解値の範囲）、z-score尺度のRMSE（RMSE÷正解値の標準偏差）。全図共通の正解データを基準とし、roll・yawの誤差は循環角度で算出。「状態：10成分全体」は正規化後の全10成分・全n問をまとめたRMSE／MAE（範囲基準・標準偏差基準の4種類）。各成分は等重みです。従来の成分別RMSEの単純平均は別分類で残しています。</p><p><a href="definitions.json">計算定義・正規化基準・出典</a> · <a href="metrics.json">集計値・有効n</a></p><nav>試合 <select id="cohort">COHORT</select> 図の種類 <select id="category">CATEGORY</select> <span id="count"></span></nav>CARDS<script>function filter(){let n=0;document.querySelectorAll('article').forEach(e=>{e.hidden=e.dataset.cohort!==document.getElementById('cohort').value||e.dataset.category!==document.getElementById('category').value;if(!e.hidden)n++});document.getElementById('count').textContent=n+' 枚'}document.querySelectorAll('select').forEach(e=>e.onchange=filter);filter();</script></html>'''
    if any(r['category'] in ('Action adjacent MAE','State change ratio (range)','State change ratio (SD)') for r in inventory):
        note = '<p>隣接値との比較：nextは予測と直前の実測値、lastは予測と直後の実測値。状態は正規化した予測変化のL2ノルム÷実測変化のL2ノルムを各問で計算して平均。0＝隣接値のコピー、1＝変化の大きさが一致（方向や正しさは保証しない）。これらの差分の符号は改善・悪化を意味しません。last-actionの直後の行動は入力に含まれていないため、その一致は入力コピーの証拠にはなりません。<a href="neighbor_definitions.json">定義・除外件数</a> · <a href="neighbor_per_query.json">問題別の変化量と比</a></p>'
        page = page.replace('<nav>', note+'<nav>')
    (output/'index.html').write_text(page.replace('COHORT',options(r['cohort'] for r in inventory)).replace('CATEGORY',options(r['category'] for r in inventory)).replace('CARDS',cards),encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='sliding_window_v1')
    parser.add_argument('--output-dir', type=Path, default=None, help='New directory; never overwrite existing plots')
    args = parser.parse_args(argv)
    source = (HERE/'results'/args.run).resolve()
    if not source.is_relative_to((HERE/'results').resolve()): parser.error('--run must be inside results')
    output = args.output_dir or source/'analysis'/('plots_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    if output.exists(): parser.error('Output already exists; choose a new --output-dir')
    data = load_data(source)
    if len(data['rows']) != 3600 or len({r['episode_id'] for r in data['rows']}) != 3:
        raise ValueError('This report expects the full three-episode 3600-query design')
    plotter = Plotter(data, output)
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.unicode_minus':False})
    definitions = dict(created_at=datetime.now().astimezone().isoformat(), plan_id=data['plan_id'],
        sources=data['sources'], normalization=plotter.scales,
        formulas=dict(action_accuracy='100 * mean(all three predicted IDs equal ground truth)',
                      action_bin_mae='mean over queries of mean over 3 components abs(predicted ID - true ID)',
                      state_mae='mean(abs(residual)) per component',
                      state_nrmse='sqrt(mean(residual**2)) / fixed reference range, per component',
                      state_zrmse='sqrt(mean(residual**2)) / fixed reference population SD (ddof=0), per component',
                      macro='arithmetic mean of the 10 component normalized RMSEs, not pooled raw units',
                      difference='metric(all) - metric(self), both recomputed on identical valid paired windows'),
        notes=['Reference uses all unique planned (episode,target,target_step) state truths across both state tasks and all H; model/scope duplicates removed.',
               'Reference scales are descriptive full-evaluation references, fixed also in per-episode plots; not theoretical physical bounds or training-fitted scales.',
               'Z-scaled RMSE applies the SAME truth mean and SD to predictions and truth; mean cancels in the residual. Predictions are not standardized separately.',
               'Roll and yaw errors are wrapped to [-180,180). Reference angles use the minimal covering arc via largest-gap cut before range/SD.',
               'Zero range/SD is undefined, not zero error; any undefined component makes the macro undefined.',
               'n is query-window count, not number of vector components or independent episodes. No uncertainty intervals or significance tests.',
               'Self/all and model comparisons use matched problems; H and task comparisons can change target time.',
               'Invalid predictions excluded from metrics; paired differences exclude a window if either scope is invalid. Individual panel and paired sample sizes can differ.',
               'Tokens use available usage regardless of parse success; missing counts are excluded, never zero-filled. Reasoning is already included in completion.',
               'Confusion matrices pool all H and report counts plus within-true-class percentages; no scope-difference confusion matrix.'])
    definitions['formulas'].update(OVERALL_FORMULAS)
    definitions['notes'].append('10D overall metrics weight all 10 normalized dimensions equally. Overall RMSE is the square root AFTER averaging squared errors across queries AND dimensions; distinct from the mean of component RMSEs. n still counts windows.')
    for cohort in ['pooled',*plotter.episodes]:
        for metric in ('action_accuracy','action_bin_mae'):
            plotter.lines(cohort,'action',metric)
        for metric in OVERALL_METRICS:
            plotter.lines(cohort,'state',metric)
        for metric in ('state_mae','state_nrmse','state_zrmse'):
            if metric != 'state_mae': plotter.lines(cohort,'state',metric)
            for dimension in range(10): plotter.lines(cohort,'state',metric,dimension)
        for dimension in range(3):
            plotter.lines(cohort,'action','action_accuracy',dimension)
            for task in ('next-action','last-action'): plotter.confusion(cohort,task,dimension)
        for kind in ('action','state'):
            for metric in ('token_prompt_tokens','token_completion_tokens','token_reasoning','token_total_tokens'):
                plotter.lines(cohort,kind,metric)
        print(f'{cohort}: {len(plotter.inventory)} figures saved',flush=True)
    for filename, content in [('definitions.json',definitions),('metrics.json',plotter.metrics),('figure_index.json',plotter.inventory)]:
        (output/filename).write_text(json.dumps(content,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    write_gallery(output,plotter.inventory)
    print(f'Saved {len(plotter.inventory)} figures (PNG): {output / "index.html"}',flush=True)


if __name__ == '__main__':
    main()
