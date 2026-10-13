"""Offline gallery: recorded telemetry overlaid with saved joint forecasts."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(HERE), str(HERE.parent / '13_LAG'), str(ROOT / 'upstream/LLM-Xavier')]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd

from lag_data.dataset import PROCESSED, load_episode, index_records
from lag_data.definitions import ACTIONS, ACTION_LABELS, STATE_FEATURES, STATE_UNITS
from trajectory_experiment import execution as ex, evaluation

COLORS = {'sol': '#d35417', 'terra': '#157f70', 'luna': '#8055bf'}
MARKERS = {'sol': '^', 'terra': 's', 'luna': 'x'}
TRUTH = '#263e55'
HISTORY = '#e3effa'
FUTURE = '#fff2d8'


def finite(value):
    return None if pd.isna(value) or not math.isfinite(float(value)) else float(value)


def recorded_window(records, target, first, last):
    """Missing/invalid/inactive telemetry is a gap, never filled or interpolated."""
    result = []
    for step in range(first, last + 1):
        row = records.get((step, target))
        alive = row is not None and pd.notna(row['alive']) and row['alive'] == 1
        state_ok = alive and pd.notna(row['state_valid']) and row['state_valid'] == 1
        action_ok = alive and row['action_status'] == 'recorded'
        result.append(dict(step=step,
                           state=[finite(row[f]) for f in STATE_FEATURES] if state_ok else [None] * 10,
                           action=[finite(row[f'{f}_id']) for f in ACTIONS] if action_ok else [None] * 4))
    return result


def line_points(steps, values, circular=False):
    """Insert a gap at angle wraps, retaining BOTH original endpoints."""
    x, y = [], []
    previous = None
    for step, value in zip(steps, values):
        if circular and previous is not None and value is not None and abs(value - previous) > 180:
            x.append(float('nan'))
            y.append(float('nan'))
        x.append(step)
        y.append(float('nan') if value is None else value)
        previous = value
    return x, y


def predictions_from_saved(plan, source):
    """Reparse raw responses against their own sample; never trust row ordering."""
    api = source / 'api'
    identity = api / 'identity.json'
    if identity.exists() and ex.read_json(identity)['plan_id'] != plan['plan_id']:
        raise ValueError('Response identity differs from plan')
    requests = ex.read_journal(api / 'requests.jsonl')
    responses = ex.read_journal(api / 'responses.jsonl')
    if (requests or responses) and not identity.exists():
        raise ValueError('Response identity is missing')
    queries = {q['query_id']: q for q in plan['queries']}
    attempted, saved = set(), {}
    for row in requests:
        qid = row['query_id']
        if qid not in queries or qid in attempted or row['request'] != ex.request_for(plan, queries[qid]):
            raise ValueError('Invalid request journal')
        attempted.add(qid)
    for row in responses:
        qid = row['query_id']
        if qid not in attempted or qid in saved:
            raise ValueError('Unknown or duplicate response query_id')
        saved[qid] = row
    result = defaultdict(dict)
    for qid, query in queries.items():
        sid, alias = query['sample_id'], query['model_alias']
        if alias in result[sid]:
            raise ValueError('Multiple responses per sample/model are not supported')
        row = saved.get(qid)
        sample = plan['samples'][sid]
        entry = dict(query_id=qid, model_alias=alias, model=query['model'], prediction=None,
                     status='pending' if qid in attempted else 'not_submitted', response_text=None)
        if row is not None:
            text = evaluation.response_text(row['raw_response'])
            metrics = evaluation.score(sample['prediction_fields'], sample['ground_truth'], text)
            entry.update(metrics, response_text=text, api_error=row.get('error_type'))
            entry['status'] = 'api_error' if row.get('error_type') else ('valid' if metrics['parse_success'] else 'parse_error')
            if entry['status'] != 'valid':
                entry['prediction'] = None
        result[sid][alias] = entry
    return result


def prepare_views(plan, source, before=50, after=10):
    ex.verify_plan(plan)
    if before < max(s['H'] for s in plan['samples'].values()) or after < 0:
        raise ValueError('before must cover the complete history; after must be nonnegative')
    predictions = predictions_from_saved(plan, source)
    records = {}
    for episode in {s['episode_id'] for s in plan['samples'].values()}:
        frame = load_episode(PROCESSED / f'{episode}.csv')
        if any(s['csv_sha256'] != frame.attrs['csv_sha256'] for s in plan['samples'].values() if s['episode_id'] == episode):
            raise ValueError('CSV differs from saved plan')
        records[episode] = index_records(frame)
    views = []
    for sid, sample in sorted(plan['samples'].items(), key=lambda p: tuple(
        p[1][key] for key in ('episode_id', 'target_id', 'sample_number', 'H', 'observation_scope'))):
        t = sample['target_step']
        expected_steps = list(range(t, t + sample['horizon']))
        if sample['task'] != 'trajectory' or sample['target_steps'] != expected_steps:
            raise ValueError('Expected a contiguous trajectory target')
        if sample['history_steps'] != list(range(t - sample['H'], t)):
            raise ValueError('History does not end immediately before the forecast')
        window = recorded_window(records[sample['episode_id']], sample['target_id'], t - before, expected_steps[-1] + after)
        by_step = {r['step']: r for r in window}
        expected_truth = [dict(step=step, state=by_step[step]['state'], action=by_step[step]['action'][:3]) for step in expected_steps]
        if expected_truth != sample['ground_truth']:
            raise ValueError('Timeline truth differs from saved ground truth')
        views.append(dict(sample_id=sid, **{key: sample[key] for key in (
            'episode_id', 'target_id', 'sample_number', 'H', 'horizon', 'observation_scope',
            'history_steps', 'target_steps', 'target_step', 'ground_truth')},
            window=window, models=predictions[sid],
            images={mode: f'figures/{sid}_{mode}.png' for mode in ('context', 'forecast')}))
    return views


def draw(view, output, mode):
    t, last = view['target_step'], view['target_steps'][-1]
    first = view['window'][0]['step'] if mode == 'context' else t - 1
    end = view['window'][-1]['step'] if mode == 'context' else last
    window = [row for row in view['window'] if first <= row['step'] <= end]
    x = [row['step'] - t for row in window]
    fig, axes = plt.subplots(7, 2, figsize=(16, 17), sharex=True)
    fig.subplots_adjust(left=.08, right=.985, bottom=.07, top=.855, hspace=.53, wspace=.24)
    scope = 'self' if view['observation_scope'] == 'self' else 'all aircraft + missiles'
    fig.suptitle(f"{view['target_id']} | question {view['sample_number']+1} | H={view['H']} | {scope}", y=.982, fontsize=18, weight='bold')
    fig.text(.5, .956, f"{view['episode_id']} | forecast step {t}..{last} | " +
             ('history and surrounding telemetry' if mode == 'context' else '10-step forecast detail (plus last observed step)'),
             ha='center', fontsize=11)
    handles = [Line2D([], [], color=TRUTH, lw=2, marker='o', markersize=3, label='Recorded truth')]
    for alias, entry in view['models'].items():
        suffix = '' if entry['status'] == 'valid' else f" [{entry['status']}; not plotted]"
        handles.append(Line2D([], [], color=COLORS[alias], ls='--', marker=MARKERS[alias],
                              markersize=5, label=alias + suffix))
    handles += [Patch(facecolor=HISTORY, label='History supplied to model'), Patch(facecolor=FUTURE, label='Hidden forecast interval')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .943), ncol=3, frameon=False, fontsize=10)
    panels = [('state', i, field) for i, field in enumerate(STATE_FEATURES)] + [('action', i, field) for i, field in enumerate(ACTIONS)]
    for ax, (kind, index, field) in zip(axes.flat, panels):
        ax.axvspan(view['history_steps'][0] - t - .5, -.5, color=HISTORY, zorder=0)
        ax.axvspan(-.5, last - t + .5, color=FUTURE, zorder=0)
        ax.axvline(-.5, color='#a78c59', ls=':', lw=1)
        values = [row[kind][index] for row in window]
        circular = field in ('roll_deg', 'yaw_deg')
        if kind == 'state':
            xx, yy = line_points(x, values, circular)
            ax.plot(xx, yy, color=TRUTH, lw=1.8, zorder=3)
            ax.scatter(x, values, color=TRUTH, s=10, zorder=3)
            ax.set_ylabel(STATE_UNITS[field], fontsize=9)
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        else:
            y = [float('nan') if value is None else value for value in values]
            ax.step(x, y, where='post', color=TRUTH, lw=1.8, zorder=3)
            ax.scatter(x, y, color=TRUTH, s=12, zorder=3)
            labels = ACTION_LABELS[field]
            ax.set_yticks(range(len(labels)), [f'{i}: {label}' for i, label in enumerate(labels)], fontsize=8)
            ax.set_ylim(-.35, len(labels)-.65)
        if field != 'shoot_action':
            for alias, entry in view['models'].items():
                if entry['status'] != 'valid':
                    continue
                pred = entry['prediction']
                px = [r['step']-t for r in pred]
                py = [r[kind][index] for r in pred]
                if kind == 'state':
                    xx, yy = line_points(px, py, circular)
                    ax.plot(xx, yy, color=COLORS[alias], ls='--', lw=1.3, alpha=.9, zorder=4)
                else:
                    ax.step(px, py, where='post', color=COLORS[alias], ls='--', lw=1.3, alpha=.9, zorder=4)
                ax.scatter(px, py, marker=MARKERS[alias], s=24, linewidths=1,
                           facecolors='none' if alias != 'luna' else COLORS[alias],
                           edgecolors=COLORS[alias] if alias != 'luna' else None, zorder=5)
        ax.set_title(field + ('  (recorded only; not predicted)' if field == 'shoot_action' else ''), loc='left', fontsize=11)
        ax.grid(alpha=.17)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlim(first-t-.5, end-t+.5)
        ticks = list(range(-1, 10)) if mode == 'forecast' else sorted(set([first-t, -40, -30, -20, -10, 0, 9, end-t]))
        ax.set_xticks([v for v in ticks if first-t <= v <= end-t])
        ax.tick_params(axis='x', labelbottom=True, labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('Step relative to first forecast t (1 step = 0.2 s)', fontsize=10)
    fig.text(.5, .018, 'Forecast points 0..9 are aligned with the same recorded steps. No truth-to-prediction connecting line.\n'
             'Only the blue history was provided to the model. Gaps = missing / invalid / inactive. '
             'Roll/yaw lines break at wraps; values are unchanged.', ha='center', fontsize=9, color='#526477')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=110, facecolor='white')
    plt.close(fig)


def build(source, output, *, before=50, after=10):
    if output.exists():
        raise FileExistsError('Choose a new --output-dir; existing galleries are not overwritten')
    plan = ex.read_json(source / 'manifest.json')
    views = prepare_views(plan, source, before, after)
    status = Counter(m['status'] for v in views for m in v['models'].values())
    metadata = dict(plan_id=plan['plan_id'], scoring_version=evaluation.SCORING_VERSION,
                    views=len(views), figures=2*len(views), responses=dict(status),
                    before=before, after=after, dt_s=.2,
                    hashes={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in (source/'manifest.json', source/'api/responses.jsonl',
                                         Path(__file__), HERE/'templates/trajectory_timelines.html', Path(evaluation.__file__)) if path.exists()})
    output.mkdir(parents=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.unicode_minus': False})
    for index, view in enumerate(views, 1):
        for mode, relative in view['images'].items():
            draw(view, output / relative, mode)
        print(f'{index}/{len(views)} conditions rendered', flush=True)
    ex.write_json(output / 'timeline_index.json', views)
    ex.write_json(output / 'metadata.json', metadata)
    payload = json.dumps(dict(views=views, metadata=metadata), ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    template = (HERE / 'templates/trajectory_timelines.html').read_text()
    (output / 'index.html').write_text(template.replace('__DATA__', payload), encoding='utf-8')
    print(f"Saved: {output/'index.html'}\nResponses: {dict(status)}", flush=True)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='trajectory_batch_v1')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--before', type=int, default=50)
    parser.add_argument('--after', type=int, default=10, help='Recorded steps after the ten-step forecast')
    args = parser.parse_args(argv)
    source = (HERE / 'results' / args.run).resolve()
    if not source.is_relative_to((HERE / 'results').resolve()) or source == (HERE / 'results').resolve():
        parser.error('--run must identify a directory inside results')
    output = args.output_dir or source / 'analysis/trajectory_timelines_v1'
    build(source, output.resolve(), before=args.before, after=args.after)


if __name__ == '__main__':
    main()
