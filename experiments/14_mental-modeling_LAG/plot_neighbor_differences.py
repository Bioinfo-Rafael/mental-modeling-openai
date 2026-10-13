"""Append adjacent-action MAE and state-change ratio figures, offline only."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import plot_results as plots
from build_results_viewer import HERE, load_data
from plot_question_timelines import recorded_window
from lag_data.dataset import PROCESSED, load_episode, index_records


def anchor_step(task, target_step):
    return target_step - 1 if task.startswith('next-') else target_step + 1


def state_ratio(prediction, truth, neighbor, scale):
    scale = np.asarray(scale, float)
    if np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError('All ten normalization scales must be finite and positive')
    actual = float(np.linalg.norm(plots.state_errors(truth, neighbor) / scale))
    predicted = float(np.linalg.norm(plots.state_errors(prediction, neighbor) / scale))
    return dict(actual_change=actual, predicted_change=predicted,
                ratio=predicted/actual if actual != 0 else None,
                zero_actual_change=actual == 0, exact_neighbor_copy=predicted == 0)


def attach_metrics(data, scales):
    episode_records, hashes = {}, {}
    for ep in sorted({r['episode_id'] for r in data['rows']}):
        frame = load_episode(PROCESSED/f'{ep}.csv')
        if any(s['csv_sha256'] != frame.attrs['csv_sha256'] for s in data['samples'].values() if s['episode_id']==ep):
            raise ValueError('Source CSV differs from frozen experiment')
        hashes[ep] = frame.attrs['csv_sha256']
        episode_records[ep] = index_records(frame)
    details = []
    for r in data['rows']:
        sample = data['samples'][r['sample_id']]
        kind = 'action' if r['task'].endswith('action') else 'state'
        offset = -1 if r['task'].startswith('next-') else 1
        t = r['target_step']
        window = recorded_window(episode_records[r['episode_id']],r['target_id'],t,1,1)
        width = 3 if kind=='action' else 10
        truth = window[1][kind][:width]
        neighbor = window[1+offset][kind][:width]
        if truth != sample['ground_truth']:
            raise ValueError('Actual target differs from frozen truth')
        predicted = (r['scoring'] or {}).get('prediction')
        keys = ['action_neighbor_mae'] if kind=='action' else ['state_change_ratio_range','state_change_ratio_sd']
        r['neighbor_metrics'] = dict.fromkeys(keys)
        detail = dict(query_id=r['query_id'], comparison_id=r['comparison_id'], sample_id=r['sample_id'],
                      **{k:r[k] for k in ('episode_id','target_id','model_alias','H','task','observation_scope','sample_number')},
                      target_step=t, neighbor_step=anchor_step(r['task'],t), truth=truth,
                      neighbor=neighbor, prediction=predicted, neighbor_in_prompt=r['task']!='last-action')
        if any(v is None for v in neighbor):
            detail['excluded_reason']='missing_neighbor'
        elif predicted is None:
            detail['excluded_reason']='invalid_prediction'
        elif kind=='action':
            r['neighbor_metrics']['action_neighbor_mae'] = float(np.mean(np.abs(np.asarray(predicted)-neighbor)))
            detail.update(excluded_reason=None,exact_neighbor_copy=predicted==neighbor,
                          actual_neighbor_mae=float(np.mean(np.abs(np.asarray(truth)-neighbor))))
        else:
            detail['state_norms']={}
            for scale_name in ('range','sd'):
                result=state_ratio(predicted,truth,neighbor,scales[scale_name])
                detail['state_norms'][scale_name]=result
                r['neighbor_metrics'][f'state_change_ratio_{scale_name}']=result['ratio']
            detail.update(excluded_reason='zero_actual_change' if result['zero_actual_change'] else None,
                          exact_neighbor_copy=result['exact_neighbor_copy'])
        detail['metrics']=r['neighbor_metrics']
        details.append(detail)
    return details,hashes


def atomic_json(path, data):
    temp=path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='sliding_window_v1')
    parser.add_argument('--output-dir',type=Path,help='Existing comparison report or a new directory')
    args=parser.parse_args(argv)
    source=(HERE/'results'/args.run).resolve()
    if not source.is_relative_to((HERE/'results').resolve()):parser.error('--run must be inside results')
    output=args.output_dir or source/'analysis'/'comparison_plots_v1'
    if (output/'neighbor_definitions.json').exists():parser.error('Neighbor report already exists; choose a new output directory')
    data=load_data(source)
    old_index=[];old_metrics=[]
    if output.exists():
        defs=json.loads((output/'definitions.json').read_text())
        if defs['plan_id']!=data['plan_id'] or defs['sources']!=data['sources']:
            raise ValueError('Existing report uses different source data')
        old_index=json.loads((output/'figure_index.json').read_text())
        old_metrics=json.loads((output/'metrics.json').read_text())
    scales=plots.reference_scales(data['samples'])
    if old_index and defs['normalization']!=scales:raise ValueError('Normalization differs from existing figures')
    details,hashes=attach_metrics(data,scales)
    plotter=plots.Plotter(data,output)
    for cohort in ['pooled',*plotter.episodes]:
        for metric,(_,_,category) in plots.NEIGHBOR_METRICS.items():
            kind=metric.split('_')[0]
            file=output/plots.figure_directory(cohort,category,f'{kind}_{metric}')/f'{kind}_{metric}.png'
            if file.exists():raise FileExistsError(file)
    output.mkdir(parents=True,exist_ok=True)
    plots.plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.unicode_minus':False})
    for cohort in ['pooled',*plotter.episodes]:
        for metric in plots.NEIGHBOR_METRICS:
            plotter.lines(cohort,metric.split('_')[0],metric)
    definitions=dict(created_at=datetime.now().astimezone().isoformat(),plan_id=data['plan_id'],sources=data['sources'],
                     csv_sha256=hashes,normalization=scales,
                     formulas=dict(action_next='mean_i mean_j abs(prediction[t,j] - actual_action[t-1,j]); j=3 movement IDs',
                                   action_last='mean_i mean_j abs(prediction[t,j] - actual_action[t+1,j]); j=3 movement IDs',
                                   state_next='mean_i ||wrap(prediction[t] - S[t-1]) / scale||_2 / ||wrap(S[t] - S[t-1]) / scale||_2',
                                   state_last='mean_i ||wrap(S[t+1] - prediction[t]) / scale||_2 / ||wrap(S[t+1] - S[t]) / scale||_2',
                                   difference='mean ratios/MAEs for all - self, both computed over the same valid paired query IDs'),
                     notes=['t denotes the prediction target_step, not next-state query_step (which is t-1).',
                            'State uses a 10-dimensional L2 norm after shared fixed range or SD normalization. Same scales as preceding figures; angles use shortest roll/yaw residuals.',
                            'Ratio computed per query, then arithmetic mean; not ratio of aggregated changes.',
                            '0 means exact neighbor state copying; 1 means equal change magnitude, not equal direction or a correct prediction.',
                            'Zero actual-change norm gives an undefined ratio (including 0/0). Excluded, counted, and retained in per-query data; no epsilon substitution or clipping.',
                            'Action uses the mean absolute difference across three movement IDs, not shoot.',
                            'For last-action, actual action at t+1 was NOT supplied in the prompt; equality cannot establish input copying. Actual t-1 action was supplied.',
                            'If actual action is unchanged, matching the neighbor can also be correct. Actual neighbor MAE retained per query to distinguish this.',
                            'No improvement/worsening direction is assigned to these metrics or the all-minus-self difference.'],
                     excluded_counts=dict(Counter(r['excluded_reason'] for r in details if r['excluded_reason'])),
                     metrics_summary={key:dict(valid=sum(r['metrics'].get(key) is not None for r in details),
                                                    eligible=sum(key in r['metrics'] for r in details)) for key in plots.NEIGHBOR_METRICS})
    if not old_index:
        atomic_json(output/'definitions.json',dict(plan_id=data['plan_id'],sources=data['sources'],normalization=scales))
    atomic_json(output/'neighbor_definitions.json',definitions)
    atomic_json(output/'neighbor_per_query.json',details)
    atomic_json(output/'figure_index.json',old_index+plotter.inventory)
    atomic_json(output/'metrics.json',old_metrics+plotter.metrics)
    plots.write_gallery(output,old_index+plotter.inventory)
    print(json.dumps(dict(figures_added=len(plotter.inventory),output=str(output/'index.html'),
                         excluded=definitions['excluded_counts'],metrics=definitions['metrics_summary']),ensure_ascii=False))


if __name__=='__main__':main()
