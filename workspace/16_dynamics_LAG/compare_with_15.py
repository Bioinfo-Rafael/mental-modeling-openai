"""Offline paired comparison: identical planned questions, common valid answers."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_results_viewer import HERE, load_data, ex
from plot_results import reference_scales, prepare, state_errors, FIELDS

OUTPUT = HERE/'results/dynamics_batch_v1/analysis/comparison_with_15_v1'
BASELINE = HERE.parent/'15_trajectory_LAG/results/trajectory_batch_v1'
TREATMENT = HERE/'results/dynamics_batch_v1'


def change(a,b):
    return {'exp15':a,'exp16':b,'delta_16_minus_15':b-a if a is not None else None,
            'reduction_pct':100*(a-b)/a if a is not None and a!=0 else None}


def pair_rows(left,right):
    def index(rows):
        pairs={(r['problem_id'],r['observation_scope'],r['model_alias']):r for r in rows}
        assert len(pairs)==len(rows), 'Duplicate comparison keys'
        return pairs
    a,b=index(left),index(right)
    assert a.keys()==b.keys(), 'Experiments must have identical planned questions'
    pairs=[]
    for key in a:
        x,y=a[key],b[key]
        for field in ('truth','target_steps','model','H','target_id','episode_id'):
            assert x[field]==y[field], field
        pairs.append((x,y))
    return pairs


def aggregate(pairs, label):
    metrics=sorted(set().union(*(r['metrics'] for p in pairs for r in p)))
    n=len(pairs)
    valid=[(x,y) for x,y in pairs if x['prediction'] is not None and y['prediction'] is not None]
    result={'group':label,'n_planned':n,'n_common_valid':len(valid),
            'n_only15_valid':sum(x['prediction'] is not None and y['prediction'] is None for x,y in pairs),
            'n_only16_valid':sum(x['prediction'] is None and y['prediction'] is not None for x,y in pairs),
            'n_neither_valid':sum(x['prediction'] is None and y['prediction'] is None for x,y in pairs),
            'metrics':{}}
    counts=[sum(p[i]['prediction'] is not None for p in pairs) for i in (0,1)]
    result['format_success']={'exp15_count':counts[0],'exp16_count':counts[1],
                              'exp15_pct':100*counts[0]/n,'exp16_pct':100*counts[1]/n,
                              'delta_pp':100*(counts[1]-counts[0])/n}
    correct=[]
    for i in (0,1):
        correct.append(sum(10-p[i]['metrics']['action_wrong_steps'] for p in pairs if p[i]['prediction'] is not None)/(n*10))
    result['valid_and_action_correct_step_rate']={'exp15_pct':100*correct[0],'exp16_pct':100*correct[1],
                                                 'delta_pp':100*(correct[1]-correct[0])}
    for name in metrics:
        # Usage comparisons include format failures when both token counts exist.
        eligible=pairs if name.startswith('token_') else valid
        pp=[(x,y) for x,y in eligible if x['metrics'].get(name) is not None and y['metrics'].get(name) is not None]
        a=np.array([x['metrics'][name] for x,y in pp]);b=np.array([y['metrics'][name] for x,y in pp])
        value=change(float(a.mean()),float(b.mean())) if len(pp) else change(None,None)
        value.update(n_pairs=len(pp),n_improved=int((b<a-1e-12).sum()),
                     n_worsened=int((b>a+1e-12).sum()),n_tied=int((np.abs(b-a)<=1e-12).sum()))
        result['metrics'][name]=value
    horizons=[]
    scale=np.array(SCALES['range'])
    for h in range(10):
        estimates=[]
        for i in (0,1):
            residual=np.array([state_errors(np.array(p[i]['prediction'][h]['state']),np.array(p[i]['truth'][h]['state'])) for p in valid])
            if len(valid):
                norm=np.abs(residual)/scale
                wrong=[p[i]['prediction'][h]['action']!=p[i]['truth'][h]['action'] for p in valid]
                estimates.append({'state_nmae':float(norm.mean()),'action_wrong_rate':float(np.mean(wrong)),
                                  **{f'mae.{f}':float(np.abs(residual[:,j]).mean()) for j,f in enumerate(FIELDS)}})
        horizons.append({'horizon':h+1,'time_ahead_s':(h+1)*.2,'n_pairs':len(valid),
                         'metrics':{k:change(estimates[0][k],estimates[1][k]) for k in estimates[0]} if estimates else {}})
    result['horizons']=horizons
    if valid:
        for i,key in enumerate(('exp15','exp16')):
            assert np.isclose(sum(h['metrics']['state_nmae'][key] for h in horizons),result['metrics']['state_sum_nmae'][key])
    return result


def make_plot(groups):
    fig,axs=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    for col,scope in enumerate(('self','all_aircraft_missiles')):
        selected=[g for g in groups if g['group']['target']=='all' and g['group']['scope']==scope and g['group']['model']!='all']
        for row,metric in enumerate(('state_sum_nmae','action_wrong_steps')):
            ax=axs[row,col];pos=np.arange(len(selected));width=.34
            for offset,experiment,color in [(-width/2,'exp15','#4779ac'),(width/2,'exp16','#e45b22')]:
                values=[g['metrics'][metric][experiment] for g in selected]
                bars=ax.bar(pos+offset,values,width,color=color,label=experiment)
                ax.bar_label(bars,fmt='%.3g',padding=3,fontsize=9)
            ax.set_xticks(pos,[g['group']['model']+f'\npaired n={g["n_common_valid"]}/{g["n_planned"]}' for g in selected])
            ax.set_title(scope);ax.set_ylabel('Sum of 10 normalized state MAEs' if row==0 else 'Incorrect action steps / 10')
            ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            ax.margins(y=.2)
    axs[0,0].legend()
    fig.suptitle('Experiment 15 vs 16 | paired valid responses only',fontsize=17)
    fig.supxlabel('Lower is better. Same windows, truth and normalization. Parse failures excluded from accuracy; reported separately.',fontsize=10)
    fig.savefig(OUTPUT/'paired_comparison.png',dpi=150);plt.close(fig)


def main():
    global SCALES
    if OUTPUT.exists(): raise FileExistsError('Comparison already exists; preserve this result or select a new output version.')
    old,new=load_data(BASELINE),load_data(TREATMENT)
    assert old['request_options']==new['request_options']
    SCALES=reference_scales(old['samples'])
    other_scales=reference_scales(new['samples'])
    # Different sample-ID ordering can change floating-point reduction order.
    # Pairing below verifies the actual truth exactly; use the baseline scales
    # for both experiments, after checking only roundoff differences exist.
    for key in SCALES:
        if key in ('minimum','maximum','mean','range','sd'):
            assert np.allclose(SCALES[key],other_scales[key],rtol=1e-12,atol=1e-12)
        else:
            assert SCALES[key]==other_scales[key]
    pairs=pair_rows(prepare(old,SCALES),prepare(new,SCALES))
    groups=[]
    for target in ['all','A0100','A0200']:
        for model in ['all','sol','terra','luna']:
            for scope in ['all','self','all_aircraft_missiles']:
                pp=[p for p in pairs if (target=='all' or p[0]['target_id']==target)
                    and (model=='all' or p[0]['model_alias']==model)
                    and (scope=='all' or p[0]['observation_scope']==scope)]
                groups.append(aggregate(pp,dict(target=target,model=model,scope=scope)))
    OUTPUT.mkdir(parents=True)
    payload={'sources':{'exp15':old['sources'],'exp16':new['sources']},'normalization':SCALES,
             'definitions':{'pair_key':'problem_id + observation_scope + model_alias',
                 'delta':'exp16 - exp15; negative means lower error or token usage.',
                 'reduction_pct':'100 * (paired mean exp15 - paired mean exp16) / paired mean exp15; unavailable if baseline is zero.',
                 'state':'Same sum-over-10 metrics as view_results; shared range/SD from 100 unique planned target states; roll/yaw use shortest-angle errors.',
                 'accuracy_denominator':'Both experiments must have a valid full 10-step answer. Each response window has equal weight.',
                 'usage_denominator':'Both usage values available, including parse failures.',
                 'end_to_end_action':'Count steps with both a valid full response and all three correct movement IDs / all planned steps. This is a success rate, not state-error imputation.',
                 'interpretation':'Descriptive paired results only; few dependent windows and a single recording. The empirical prior in 16 was fitted using the evaluated recording.'},
             'groups':groups}
    ex.write_json(OUTPUT/'metrics.json',payload)
    ex.write_json(OUTPUT/'paired_windows.json',[{'problem_id':x['problem_id'],'model':x['model_alias'],'scope':x['observation_scope'],
                'target':x['target_id'],'sample_number':x['sample_number'],'query_id15':x['query_id'],'query_id16':y['query_id'],
                'status15':x['status'],'status16':y['status'],'metrics15':x['metrics'],'metrics16':y['metrics']} for x,y in pairs])
    rows=[];hh=[]
    for group in groups:
        for name,m in group['metrics'].items():rows.append(dict(**group['group'],metric=name,n_planned=group['n_planned'],**m))
        for h in group['horizons']:
            for name,m in h['metrics'].items():hh.append(dict(**group['group'],metric=name,horizon=h['horizon'],time_ahead_s=h['time_ahead_s'],n_pairs=h['n_pairs'],**m))
    for filename,rr in [('metrics.csv',rows),('horizons.csv',hh)]:
        with (OUTPUT/filename).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rr[0]));writer.writeheader();writer.writerows(rr)
    make_plot(groups)
    overall=groups[0]
    lines=['Experiment 15 → 16 比較','',
           '同じ問題・入力範囲・モデルを対応付け、精度は両実験で全10stepが有効な回答だけで比較。',
           '全計画の形式成功率とtoken量は形式エラーを含めて別集計。',
           f'形式成功率: {overall["format_success"]}',f'共通有効回答: {overall["n_common_valid"]}/{overall["n_planned"]}',
           f'15のみ有効: {overall["n_only15_valid"]}, 16のみ有効: {overall["n_only16_valid"]}, 両方無効: {overall["n_neither_valid"]}',
           '', '主要指標（誤差は小さい方が良い。reduction_pctは正なら改善）:']
    for metric in ['state_sum_nmae','state_sum_nrmse','state_sum_zmae','action_wrong_steps','action_sum_bin_error',
                   'state_sae.roll_deg','state_sae.pitch_deg','state_sae.yaw_deg','token_prompt_tokens','token_total_tokens']:
        lines.append(metric+': '+json.dumps(overall['metrics'][metric],ensure_ascii=False))
    lines+=['','形式有効かつ移動行動3成分が正解のstep数 / 全計画step数:',json.dumps(overall['valid_and_action_correct_step_rate']),
            '', 'モデル別・入力範囲別の共通有効比較:']
    for group in groups:
        if group['group']['target']=='all' and group['group']['model']!='all' and group['group']['scope']!='all':
            lines.append(json.dumps({'group':group['group'],'n':group['n_common_valid'],'format_success':group['format_success'],
                                    'state_sum_nmae':group['metrics']['state_sum_nmae'],'action_wrong_steps':group['metrics']['action_wrong_steps']},ensure_ascii=False))
    lines+=['','state_saeは10stepの絶対誤差合計。通常の1step平均MAEに直す場合は10で割る。',
            'roll/yawは最短角度差。異なる状態単位の生誤差は混ぜず、共通正解の範囲または標準偏差で正規化する。',
            '改善率は対応する平均値の比であり、問題ごとの改善率の平均ではない。ベースライン0の場合は算出しない。',
            '予測誤差の改善は、両実験で有効な回答に条件付けた結果。形式失敗の増減を含む総合的な改善とは区別する。',
            '同じ10窓をモデル・scope間で共有するため、60回答や各時刻を独立標本として有意性を主張しない。',
            '16の係数は対象試合を含めて推定されており、未知の試合への汎化評価ではない。',
            '詳細: metrics.json / metrics.csv / horizons.csv / paired_windows.json / paired_comparison.png']
    (OUTPUT/'summary.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:19]));print('Saved:',OUTPUT)


if __name__=='__main__':main()
