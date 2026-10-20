"""Offline comparison plots: sum ten forecast-point errors, then average windows."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import hashlib
import html
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_results_viewer import load_data, HERE, ex

MODELS = ('sol', 'terra', 'luna')
COLORS = {'sol': '#B84B96', 'terra': '#2875B5', 'luna': '#DF871E'}
SCOPES = ('self', 'all_aircraft_missiles')
FIELDS = ('lon_deg', 'lat_deg', 'alt_m', 'roll_deg', 'pitch_deg', 'yaw_deg',
          'tas_mps', 'vn_mps', 've_mps', 'vu_mps')
ACTION_FIELDS = ('altitude_action', 'heading_action', 'speed_action')
UNITS = ('deg', 'deg', 'm', 'deg', 'deg', 'deg', 'm/s', 'm/s', 'm/s', 'm/s')
CIRCULAR = (3, 5)
OVERALL = ('state_sum_nmae', 'state_sum_zmae', 'state_sum_nrmse', 'state_sum_zrmse')
CATEGORY_LABELS = {'state_overall': '状態：10成分全体', 'state_components': '状態：成分別の10地点誤差合計',
                   'action_overall': '行動：10地点全体', 'action_components': '行動：成分別',
                   'confusion': '行動：混同行列', 'tokens': 'token数'}


def state_errors(prediction, truth):
    result = np.asarray(prediction, float) - np.asarray(truth, float)
    result[..., list(CIRCULAR)] = (result[..., list(CIRCULAR)] + 180) % 360 - 180
    return result


def reference_scales(samples):
    """Fixed planned truth reference; repeated model/scope states count only once."""
    unique = {}
    for sample in samples.values():
        for row in sample['ground_truth']:
            key = sample['episode_id'], sample['target_id'], row['step']
            if key in unique and unique[key] != row['state']:
                raise ValueError('Conflicting state truth')
            unique[key] = row['state']
    values = np.asarray(list(unique.values()), float)
    origins = {}
    for i in CIRCULAR:
        angles = np.unique(values[:, i] % 360)
        gaps = np.diff(np.r_[angles, angles[0] + 360])
        origin = angles[(int(np.argmax(gaps)) + 1) % len(angles)]
        values[:, i] = (values[:, i] - origin) % 360 + origin
        origins[FIELDS[i]] = float(origin)
    return dict(n_unique_target_states=len(unique), fields=list(FIELDS),
                minimum=values.min(axis=0).tolist(), maximum=values.max(axis=0).tolist(),
                mean=values.mean(axis=0).tolist(), range=np.ptp(values, axis=0).tolist(),
                sd=values.std(axis=0, ddof=0).tolist(), circular_origins_deg=origins)


def window_metrics(prediction, truth, scales):
    """Each scalar is a SUM across ten steps, never their mean or signed sum."""
    if len(prediction) != 10 or [r['step'] for r in prediction] != [r['step'] for r in truth]:
        raise ValueError('Expected all ten aligned forecast steps')
    pa, ga = (np.asarray([r['action'] for r in rows], float) for rows in (prediction, truth))
    ps, gs = (np.asarray([r['state'] for r in rows], float) for rows in (prediction, truth))
    if pa.shape != (10, 3) or ga.shape != (10, 3) or ps.shape != (10, 10) or gs.shape != (10, 10):
        raise ValueError('Invalid trajectory shape')
    residual = state_errors(ps, gs)
    metrics = dict(action_wrong_steps=float(np.any(pa != ga, axis=1).sum()),
                   action_sum_bin_error=float(np.abs(pa-ga).mean(axis=1).sum()))
    for i, field in enumerate(ACTION_FIELDS):
        metrics[f'action_wrong_steps.{field}'] = float((pa[:, i] != ga[:, i]).sum())
        metrics[f'action_sum_bin_error.{field}'] = float(np.abs(pa[:, i]-ga[:, i]).sum())
    for i, field in enumerate(FIELDS):
        metrics[f'state_sae.{field}'] = float(np.abs(residual[:, i]).sum())
    for scale, prefix in (('range', 'n'), ('sd', 'z')):
        denominator = np.asarray(scales[scale], float)
        normalized = np.divide(residual, denominator, out=np.full_like(residual, np.nan), where=denominator > 0)
        for i, field in enumerate(FIELDS):
            metrics[f'state_{prefix}sae.{field}'] = float(np.abs(normalized[:, i]).sum()) if denominator[i] > 0 else None
        for kind in ('mae', 'rmse'):
            point_errors = (np.abs(normalized).mean(axis=1) if kind == 'mae'
                            else np.sqrt(np.square(normalized).mean(axis=1)))
            metrics[f'state_sum_{prefix}{kind}'] = float(point_errors.sum()) if np.isfinite(point_errors).all() else None
    return metrics


def prepare(data, scales):
    result = []
    for row in data['rows']:
        sample = data['samples'][row['sample_id']]
        if row['H'] != 20 or sample['horizon'] != 10 or sample['task'] != 'trajectory':
            raise ValueError('This report expects H=20 and ten-step joint forecasts')
        scoring = row['scoring'] or {}
        prediction = scoring.get('prediction') if scoring.get('parse_success') else None
        metrics = window_metrics(prediction, sample['ground_truth'], scales) if prediction is not None else {}
        usage = (row['raw_response'] or {}).get('usage') or {}
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            metrics['token_' + key] = usage.get(key)
        metrics['token_reasoning'] = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens')
        result.append(dict(row, truth=sample['ground_truth'], prediction=prediction, metrics=metrics))
    return result


def estimate(rows, metric):
    valid = [r for r in rows if r['metrics'].get(metric) is not None]
    value = float(np.mean([r['metrics'][metric] for r in valid])) if valid else None
    return dict(value=value, n_valid=len(valid), n_planned=len(rows),
                comparison_ids=[r['comparison_id'] for r in valid])


def paired_estimate(left, right, metric):
    a, b = ({r['comparison_id']: r for r in rows} for rows in (left, right))
    if set(a) != set(b) or len(a) != len(left) or len(b) != len(right):
        raise ValueError('Input scopes must contain identical unique windows')
    if any(a[k]['truth'] != b[k]['truth'] for k in a):
        raise ValueError('Paired ground truth differs')
    ids = [k for k in a if a[k]['metrics'].get(metric) is not None and b[k]['metrics'].get(metric) is not None]
    x, y = (float(np.mean([group[k]['metrics'][metric] for k in ids])) if ids else None for group in (a, b))
    return dict(value=y-x if ids else None, n_valid=len(ids), n_planned=len(a),
                paired_self=x, paired_all=y, comparison_ids=ids)


def specifications():
    yield 'action_wrong_steps', 'action_overall', 'Action: incorrect steps summed over 10 points', 'Incorrect steps / window'
    yield 'action_sum_bin_error', 'action_overall', 'Action: sum of mean absolute ID errors over 10 points', 'Cumulative ID error (mean of 3 IDs)'
    for metric in OVERALL:
        scale = 'range' if '_n' in metric else 'SD'
        kind = 'MAE' if metric.endswith('mae') else 'RMSE'
        yield metric, 'state_overall', f'State: sum of 10 per-step {kind}s | {scale}-normalized 10D', 'Cumulative dimensionless error'
    for field in FIELDS:
        for prefix, label in (('state_sae', 'raw units'), ('state_nsae', 'range-normalized'), ('state_zsae', 'SD-normalized')):
            unit = UNITS[FIELDS.index(field)] if prefix == 'state_sae' else 'dimensionless'
            yield f'{prefix}.{field}', 'state_components', f'{field}: sum of 10 absolute errors | {label}', f'Cumulative absolute error ({unit})'
    for field in ACTION_FIELDS:
        yield f'action_wrong_steps.{field}', 'action_components', f'{field}: incorrect steps summed over 10 points', 'Incorrect steps / window'
        yield f'action_sum_bin_error.{field}', 'action_components', f'{field}: sum of 10 absolute ID errors', 'Cumulative ID error (bins)'
    for key in ('prompt_tokens', 'completion_tokens', 'reasoning', 'total_tokens'):
        yield 'token_' + key, 'tokens', key.replace('_', ' ').title() + ' per request', 'Tokens / request'


class Plotter:
    def __init__(self, data, output):
        self.data, self.output = data, output
        self.scales = reference_scales(data['samples'])
        self.rows = prepare(data, self.scales)
        self.inventory, self.aggregates = [], []
        self.cohorts = ['pooled', *sorted({r['target_id'] for r in self.rows})]

    def selected(self, cohort, model, scope):
        return [r for r in self.rows if (cohort == 'pooled' or r['target_id'] == cohort)
                and r['model_alias'] == model and r['observation_scope'] == scope]

    def save(self, fig, cohort, category, name, title):
        relative = Path(category) / cohort / (name.replace('.', '_') + '.png')
        (self.output / relative).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.output / relative, dpi=120, facecolor='white')
        plt.close(fig)
        self.inventory.append(dict(cohort=cohort, category=category, name=name, title=title, png=str(relative)))

    def comparison(self, cohort, metric, category, title, unit):
        fig, axes = plt.subplots(1, 3, figsize=(14, 5.3))
        fig.subplots_adjust(left=.08, right=.98, top=.76, bottom=.25, wspace=.37)
        fig.suptitle(title, fontsize=15, y=.97)
        cohort_name = 'A0100 + A0200 pooled' if cohort == 'pooled' else cohort
        aggregation = 'mean usage per request (no temporal sum)' if metric.startswith('token_') else 'mean across windows AFTER summing 10 forecast points'
        fig.text(.5, .885, f'{cohort_name} | H=20 | {aggregation}', ha='center', fontsize=10)
        for i, model in enumerate(MODELS):
            left, right = [self.selected(cohort, model, scope) for scope in SCOPES]
            records = [estimate(left, metric), estimate(right, metric), paired_estimate(left, right, metric)]
            for j, (scope, record) in enumerate(zip((*SCOPES, 'all_minus_self'), records)):
                self.aggregates.append(dict(cohort=cohort, model=model, scope=scope, metric=metric, **record))
                v, n = record['value'], record['n_valid']
                ax = axes[j]
                if v is None:
                    ax.text(i, 0, 'N/A\nn=0', ha='center', va='bottom', fontsize=9)
                else:
                    ax.bar(i, v, width=.56, color=COLORS[model], alpha=.85)
                    ax.annotate(f'{v:.3g}\nn={n}/{record["n_planned"]}', (i, v), xytext=(0, 5 if v >= 0 else -5),
                                textcoords='offset points', ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)
        for j, label in enumerate(('Self', 'All aircraft + missiles', 'All minus self (paired)')):
            ax = axes[j]
            ax.set_title(label, fontsize=11)
            ax.set_xticks(range(3), MODELS)
            ax.set_xlim(-.6, 2.6)
            ax.set_ylabel(unit if j < 2 else 'Difference (' + ('tokens' if metric.startswith('token') else 'same units') + ')', fontsize=9)
            ax.grid(axis='y', alpha=.2)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
            if j < 2:
                ax.set_ylim(0, max(ax.get_ylim()[1]*1.28, 1e-12))
            else:
                ax.axhline(0, color='#536271', lw=.9)
                low, high = ax.get_ylim()
                margin = max(high-low, 1e-10)*.28
                ax.set_ylim(low-margin, high+margin)
        top = max(axes[j].get_ylim()[1] for j in (0, 1))
        for ax in axes[:2]: ax.set_ylim(0, top)
        note = ('Tokens are measured per request, not multiplied by ten. Reasoning is included in completion.'
                if metric.startswith('token') else 'Lower error is better. The 10 points are SUMMED, not averaged; n counts query windows, not points.')
        fig.text(.5, .075, note + '\nSelf/all use valid responses; differences use identical valid pairs only. '
                 'No missing-value substitution.\nRoll/yaw use shortest-angle residuals; fixed truth reference for normalization.',
                 ha='center', fontsize=8.5, color='#526171', linespacing=1.5)
        self.save(fig, cohort, category, metric, title)

    def confusion(self, cohort, dimension):
        count = (3, 5, 3)[dimension]
        fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
        fig.subplots_adjust(left=.075, right=.90, top=.84, bottom=.12, wspace=.4, hspace=.44)
        field = ACTION_FIELDS[dimension]
        title = f'{field}: confusion matrices | all 10 forecast points'
        fig.suptitle(title, y=.975, fontsize=16)
        fig.text(.5, .925, f'{cohort} | H=20 | each valid trajectory contributes 10 points', ha='center')
        for i, scope in enumerate(SCOPES):
            for j, model in enumerate(MODELS):
                rows = self.selected(cohort, model, scope)
                valid = [r for r in rows if r['prediction'] is not None]
                matrix = np.zeros((count, count), int)
                for row in valid:
                    for p, g in zip(row['prediction'], row['truth']): matrix[g['action'][dimension], p['action'][dimension]] += 1
                denominator = matrix.sum(axis=1, keepdims=True)
                percent = np.divide(100.*matrix, denominator, out=np.full(matrix.shape, np.nan), where=denominator > 0)
                ax = axes[i, j]
                im = ax.imshow(np.ma.masked_invalid(percent), vmin=0, vmax=100, cmap='Blues')
                for y in range(count):
                    for x in range(count):
                        label = f'{matrix[y,x]}\n{percent[y,x]:.0f}%' if denominator[y,0] else '0\nN/A'
                        ax.text(x, y, label, ha='center', va='center', color='white' if percent[y,x] > 55 else '#233447', fontsize=9)
                ax.set_xticks(range(count)); ax.set_yticks(range(count))
                ax.set_xlabel('Predicted ID'); ax.set_ylabel('True ID')
                ax.set_title(f'{model} | {"self" if i==0 else "all"} | n={len(valid)}/{len(rows)}', fontsize=11)
                self.aggregates.append(dict(cohort=cohort, model=model, scope=scope, metric='confusion.'+field,
                                            counts=matrix.tolist(), n_valid=len(valid), n_planned=len(rows), points=len(valid)*10))
        fig.colorbar(im, cax=fig.add_axes([.93, .25, .015, .48]), label='Within true class (%)')
        fig.text(.5, .025, 'Cells: count and row percentage. n counts windows. Ten points within a window are not independent samples.',
                 ha='center', fontsize=9)
        self.save(fig, cohort, 'confusion', 'confusion.'+field, title)


def gallery(plotter, definitions):
    cards = ''.join(f'<article data-cohort="{r["cohort"]}" data-category="{r["category"]}"><h2>{html.escape(r["title"])}</h2>'
                    f'<a href="{r["png"]}"><img loading="lazy" src="{r["png"]}" alt="{html.escape(r["title"])}"></a></article>' for r in plotter.inventory)
    cohorts = ''.join(f'<option value="{c}">{"2機体pool（各モデル・範囲で計画n=10）" if c=="pooled" else c+"（計画n=5）"}</option>' for c in plotter.cohorts)
    categories = ''.join(f'<option value="{c}">{label}</option>' for c, label in CATEGORY_LABELS.items())
    return '''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>15 LAG · 10地点の誤差合計</title><style>
body{max-width:1550px;margin:auto;padding:26px;font:15px/1.75 system-ui;background:#f4f7fb;color:#203348}h1{font-size:27px}h2{font-size:18px}nav,article,.note{background:white;border:1px solid #d5deea;border-radius:10px;padding:18px;margin:18px 0}nav{position:sticky;top:0;z-index:2;display:flex;gap:14px;align-items:center;flex-wrap:wrap}select{font:inherit;padding:7px;border:1px solid #b9c8d7;border-radius:5px}img{width:100%;height:auto}a{color:#245caa}[hidden]{display:none!important}.formula{font-family:ui-monospace,monospace;background:#edf3fa;padding:10px}</style></head><body>
<h1>実験15 · 未来10地点の誤差合計</h1><div class="note"><p>H=20・MHIのみ。各問題で未来10地点の誤差を合計し、その合計を問題間で平均します。機体別は計画n=5、2機体poolはn=10です。</p>
<p class="formula">成分別: mean_q [ sum_(t=1..10) |予測(q,t) − 正解(q,t)| ]</p>
<p>状態全体は、成分の範囲または標準偏差で正規化した後、各地点の10成分MAE／RMSEを計算し、その10地点分を合計します。成分ごとの生の異なる単位は合算しません。14と同じくroll・yawには最短角度差を用います。時間方向の平均や、符号付き誤差の相殺は行いません。</p>
<p>左＝self、中央＝全機体＋ミサイル、右＝同じ問題の有効なペアで計算した「全機体−self」。誤差の差は負が改善です。形式不正は除外し、有効nを各棒に表示します。モデルごとに有効な問題集合が異なる場合があります。混同行列は全10地点の件数、tokenは1リクエストあたりです。</p>
<p>__COUNTS__</p><p><a href="definitions.json">計算定義・正規化基準・出典</a> · <a href="metrics.json">集計値・有効な問題ID</a> · <a href="per_query.json">問題別の10地点誤差合計</a></p></div>
<nav>対象 <select id="cohort">__COHORTS__</select> 図の種類 <select id="category">__CATEGORIES__</select><span id="count"></span></nav>__CARDS__
<script>function filter(){let n=0;document.querySelectorAll('article').forEach(e=>{e.hidden=e.dataset.cohort!==document.getElementById('cohort').value||e.dataset.category!==document.getElementById('category').value;if(!e.hidden)n++});document.getElementById('count').textContent=n+'枚'}document.querySelectorAll('select').forEach(e=>e.onchange=filter);filter();</script></body></html>'''.replace('__COHORTS__', cohorts).replace('__CATEGORIES__', categories).replace('__CARDS__', cards).replace('__COUNTS__', html.escape(f"回答の状態: {definitions['response_counts']}"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='dynamics_batch_v1')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--entry', type=Path, default=HERE/'view_results.html')
    args = parser.parse_args(argv)
    source = (HERE/'results'/args.run).resolve()
    if not source.is_relative_to((HERE/'results').resolve()) or source == (HERE/'results').resolve():
        parser.error('--run must identify a results directory')
    output = (args.output_dir or source/'analysis/comparison_plots_sum10_v1').resolve()
    if output.exists(): parser.error('Choose a new --output-dir; existing figures are not overwritten')
    if args.entry.exists(): parser.error('Choose a new --entry; existing entry is not overwritten')
    data = load_data(source)
    if len(data['rows']) != 60 or {r['episode_id'] for r in data['rows']} != {'mhi_v1.3_ic100601_win'}:
        raise ValueError('Expected the MHI-only 60-request design')
    plotter = Plotter(data, output)
    for target in ('A0100', 'A0200'):
        for model in MODELS:
            for scope in SCOPES:
                group = plotter.selected(target, model, scope)
                if len(group) != 5 or len({r['comparison_id'] for r in group}) != 5:
                    raise ValueError('Expected five unique windows per model/aircraft/scope')
    output.mkdir(parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.unicode_minus':False})
    definitions = dict(created_at_jst=datetime.now(timezone(timedelta(hours=9))).isoformat(), plan_id=data['plan_id'],
        response_counts=data['counts'], sources=data['sources'], normalization=plotter.scales,
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        formulas=dict(state_sae='mean_q(sum_t(abs(residual[q,t,d])))',
            state_nsae='mean_q(sum_t(abs(residual[q,t,d]) / reference_range[d]))',
            state_zsae='mean_q(sum_t(abs(residual[q,t,d]) / reference_sd[d]))',
            state_sum_nmae='mean_q(sum_t(mean_d(abs(residual[q,t,d] / reference_range[d]))))',
            state_sum_zmae='mean_q(sum_t(mean_d(abs(residual[q,t,d] / reference_sd[d]))))',
            state_sum_nrmse='mean_q(sum_t(sqrt(mean_d((residual[q,t,d] / reference_range[d])**2))))',
            state_sum_zrmse='mean_q(sum_t(sqrt(mean_d((residual[q,t,d] / reference_sd[d])**2))))',
            action_wrong_steps='mean_q(sum_t(any_d(predicted_action[q,t,d] != true_action[q,t,d])))',
            action_sum_bin_error='mean_q(sum_t(mean_d(abs(predicted_action[q,t,d] - true_action[q,t,d]))))',
            action_components='Same time sum, with a single action component; no dimension mean.',
            difference='mean over common valid windows of metric(all) - metric(self)',
            tokens='mean available reported usage per request, regardless of prediction parse status'),
        notes=['All errors sum 10 future points BEFORE the equal-weight mean across query windows; n counts windows.',
               'RMSE is over 10 state dimensions AT EACH POINT, then summed over time, then averaged over windows. This is not sqrt(sum of squared trajectory errors).',
               'The same fixed reference uses all unique planned (episode,target,step) truth states, including questions with invalid model responses.',
               'Roll/yaw residuals use shortest angles in [-180,180); reference uses a largest-gap cut, as in Experiment 14 plots. Raw evaluator results remain unchanged.',
               'Zero reference scales yield undefined values, never zero. Different raw state units are not summed together.',
               'Malformed trajectories are excluded entirely. No partial predictions or imputed results. Invalid rows still retain available token usage.',
               'Scope differences use only common valid pairs. Unpaired panel means may have different n. No confidence intervals or independent-point claims.'])
    for cohort in plotter.cohorts:
        for metric, category, title, unit in specifications(): plotter.comparison(cohort, metric, category, title, unit)
        for dimension in range(3): plotter.confusion(cohort, dimension)
        print(f'{cohort}: {len(plotter.inventory)} figures saved', flush=True)
    per_query = [{k:r[k] for k in ('query_id','comparison_id','model_alias','episode_id','target_id','observation_scope','sample_number','target_steps','status','metrics')} for r in plotter.rows]
    for name, content in [('definitions.json', definitions), ('metrics.json', plotter.aggregates), ('per_query.json', per_query), ('figure_index.json', plotter.inventory)]:
        ex.write_json(output/name, content)
    page = gallery(plotter, definitions)
    (output/'index.html').write_text(page, encoding='utf-8')
    args.entry.parent.mkdir(parents=True, exist_ok=True)
    base = html.escape(os.path.relpath(output, args.entry.resolve().parent) + '/', quote=True)
    args.entry.write_text(page.replace('<head>', f'<head><base href="{base}">', 1), encoding='utf-8')
    print(f'Saved {len(plotter.inventory)} figures: {args.entry}', flush=True)


if __name__ == '__main__': main()
