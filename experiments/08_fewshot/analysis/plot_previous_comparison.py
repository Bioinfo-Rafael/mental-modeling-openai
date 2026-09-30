"""Compare existing 07 n=10 and 08 n=3 results. Offline, PNG only."""
import csv
import hashlib
from importlib import import_module
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from llm_x.data import Episode
scoring = import_module('experiments.08_fewshot.runner.scoring')

OUT = HERE / 'results/analysis/previous_comparison'
TASKS = ('MountainCar-v0', 'Pendulum-v1')
MODELS = ('3.5', 'sol', 'terra', 'luna')
MODEL_NAMES = {'3.5': 'GPT-3.5', 'sol': 'Sol', 'terra': 'Terra', 'luna': 'Luna'}
MODEL_COLORS = {'3.5': '#8793a3', 'sol': '#8c6bb1', 'terra': '#2b6d9f', 'luna': '#ce713d'}
PATTERN_HATCHES = {'pattern1': '', 'pattern2': '///'}
SHOT_COLORS = {
    'terra': {4: '#aecbdf', 8: '#669bc0', 12: '#2b6d9f'},
    'luna': {4: '#efd0b6', 8: '#dfa16e', 12: '#ce713d'},
}
VIEWS = ('all', 'pool_shots', 'pool_pattern', 'pool_all')
VIEW_NAMES = {'all': '1. All six few-shot conditions', 'pool_shots': '2. Shot counts pooled',
              'pool_pattern': '3. Score patterns pooled', 'pool_all': '4. Terra / Luna: all conditions pooled'}
METRICS = ('accuracy', 'mae', 'reasoning_score')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def load_records():
    catalog_path = HERE / 'data_prep/examples.json'
    score07_path = HERE.parent / '07_view_rawdata/reasoning_for_scoring_scored_astra_high.csv'
    score08_path = HERE / 'reasoning_for_scoring_scored_astra_high.csv'
    record08_path = HERE / 'results/api/records.jsonl'
    sources = {p: p.read_bytes() for p in (catalog_path, score07_path, score08_path, record08_path)}
    examples = [e for e in json.loads(sources[catalog_path]) if e['metric'] == 'next-action' and e['H'] in (5,20) and e['ordinal'] < 10]
    assert len(examples) == 160
    scores07 = read_csv(score07_path)
    scores08 = {s['query_id']: s for s in read_csv(score08_path)}
    old_sources, episodes, output = {}, {}, []

    def evaluate(row, experiment, old_score, new_score, reasoning_available):
        path = ROOT / row['episode_path']
        if path not in episodes:
            episodes[path] = Episode.load(path)
            sources[path] = path.read_bytes()
        assert episodes[path].sha256 == row['episode_sha256']
        result = scoring.score(row, row.get('assistant_text'), episodes[path])
        assert result['ground_truth'] == row['ground_truth']
        if experiment == '08':
            assert row['status'] == result['status'] and row['prediction'] == result['prediction']
        output.append(dict(experiment=experiment, query_id=row['query_id'], task=row['task'], model=row['model_alias'],
            H=row['H'], shots=row.get('shots',0), score_pattern=row.get('score_pattern','zero'), ordinal=row['ordinal'],
            episode_path=row['episode_path'], episode_sha256=row['episode_sha256'], query_index=row['query_index'],
            correct=int(result['status']=='match'), status=result['status'],
            prediction=result['prediction'], ground_truth=result['ground_truth'],
            absolute_error=result['absolute_error'], old_reasoning_score=old_score,
            reasoning_score=new_score if reasoning_available else None,
            response_available=bool(row.get('assistant_text')), reasoning_available=reasoning_available))

    for e in examples:
        source = ROOT / e['source_records']
        if source not in old_sources:
            sources[source] = source.read_bytes()
            old_sources[source] = [json.loads(line) for line in sources[source].splitlines()]
        row = old_sources[source][e['source_line']-1]
        assert row['query_id'] == e['query_id'] and row.get('assistant_text') == e['answer']
        s = scores07[e['score_csv_record']-1]
        assert (s['model'],s['Task'],s['Metrics'],int(s['H']),s['Reasoning']) == (e['model'],e['task'],e['metric'],e['H'],e['reasoning'])
        old_score = int(s['score'])
        assert old_score in (1,2,3,4) and old_score == e['reasoning_score']
        evaluate(row,'07',old_score,5-old_score,bool(s['Reasoning'].strip()))
    for row in [json.loads(line) for line in sources[record08_path].splitlines()]:
        s = scores08[row['query_id']]
        for column,key in [('model','model_alias'),('Task','task'),('H','H'),('shots','shots'),('score_pattern','score_pattern'),('ordinal','ordinal')]:
            assert s[column] == str(row[key])
        score = int(s['score']) if s['score'] else None
        assert score in (None,1,2,3,4)
        evaluate(row,'08',None,score,bool(s['Reasoning'].strip()) and score is not None)
    assert len(output) == 304 and len({r['query_id'] for r in output}) == 304
    for task in TASKS:
        for model in MODELS:
            for h in (5,20):
                group = [r for r in output if (r['experiment'],r['task'],r['model'],r['H']) == ('07',task,model,h)]
                assert sorted(r['ordinal'] for r in group) == list(range(10))
        for model in ('terra','luna'):
            for h in (5,20):
                for pattern in ('pattern1','pattern2'):
                    for shots in (4,8,12):
                        group = [r for r in output if (r['experiment'],r['task'],r['model'],r['H'],r['score_pattern'],r['shots']) == ('08',task,model,h,pattern,shots)]
                        assert sorted(r['ordinal'] for r in group) == [0,1,2]
    return output, {str(p.relative_to(ROOT)): hashlib.sha256(raw).hexdigest() for p,raw in sources.items()}


def group_specs(view):
    specs = []
    def add(experiment, model, label, x, color, pattern=None, shots=None):
        specs.append(dict(experiment=experiment,model=model,label=label,x=x,color=color,pattern=pattern,shots=shots))
    if view == 'pool_all':
        for model,x in [('terra',0),('luna',3)]:
            add('07',model,'07',x,SHOT_COLORS[model][4])
            add('08',model,'08',x+.9,MODEL_COLORS[model])
        return specs, [('Terra',.45),('Luna',3.45)]
    for x,model in enumerate(MODELS):
        add('07',model,MODEL_NAMES[model],x,MODEL_COLORS[model])
    centers = [('07: 0-shot',1.5)]
    for model,start in [('terra',5.5),('luna',13 if view == 'all' else 10.5)]:
        if view == 'all':
            for j,shots in enumerate((4,8,12)):
                for k,pattern in enumerate(('pattern1','pattern2')):
                    add('08',model,str(shots) if k==0 else '',start+j*2.05+k*.68,SHOT_COLORS[model][shots],pattern,shots)
            center=start+2.39
        elif view == 'pool_shots':
            for j,pattern in enumerate(('pattern1','pattern2')):
                add('08',model,f'P{j+1}',start+j*.85,MODEL_COLORS[model],pattern)
            center=start+.425
        else:
            for j,shots in enumerate((4,8,12)):
                add('08',model,str(shots),start+j*.9,SHOT_COLORS[model][shots],shots=shots)
            center=start+.9
        centers.append((f'08: {MODEL_NAMES[model]}',center))
    return specs,centers


def summarize(rows, task, view):
    specs,_ = group_specs(view)
    output=[]
    for spec in specs:
        group=[r for r in rows if r['task']==task and r['experiment']==spec['experiment'] and r['model']==spec['model']
               and (spec['pattern'] is None or r['score_pattern']==spec['pattern'])
               and (spec['shots'] is None or r['shots']==spec['shots'])]
        expected=20 if spec['experiment']=='07' else {'all':6,'pool_shots':18,'pool_pattern':12,'pool_all':36}[view]
        assert len(group)==expected
        # Average base-condition means equally, including H=5/20 equally.
        cells={}
        for r in group:
            cells.setdefault((r['H'],r['score_pattern'],r['shots']),[]).append(r)
        for metric,field in [('accuracy','correct'),('mae','absolute_error'),('reasoning_score','reasoning_score')]:
            cell_values=[[r[field] for r in cell if r[field] is not None] for cell in cells.values()]
            valid=sum(len(v) for v in cell_values)
            value=float(np.mean([np.mean(v) for v in cell_values])) if all(cell_values) else None
            output.append(dict(task=task,view=view,metric=metric,experiment=spec['experiment'],model=spec['model'],
                pattern=spec['pattern'],shots=spec['shots'],n=len(group),valid_n=valid,base_conditions=len(cells),
                mean=value,correct=sum(r['correct'] for r in group),missing=len(group)-valid))
    return output


def draw(ax,task,view,metric,summary,limit):
    specs,centers=group_specs(view)
    data=[r for r in summary if (r['task'],r['view'],r['metric'])==(task,view,metric)]
    for spec,row in zip(specs,data):
        value=row['mean']
        if value is None:
            ax.text(spec['x'],.1,'N/A',ha='center',transform=ax.get_xaxis_transform())
            continue
        ax.bar(spec['x'],value,width=.66 if view=='all' else .72,color=spec['color'],
               hatch=PATTERN_HATCHES.get(spec['pattern'],''),edgecolor='#ffffff',linewidth=.6,zorder=3)
        label=f'{100*value:.3g}' if metric=='accuracy' else f'{value:.2f}'
        if row['missing']:
            label+=f"\n(n={row['valid_n']})"
        offset=14 if view=='all' and spec['pattern']=='pattern2' else 4
        ax.annotate(label,(spec['x'],value),xytext=(0,offset),textcoords='offset points',ha='center',fontsize=7.5)
    if view=='all':
        ticks=[s['x']+.34 if s['experiment']=='08' else s['x'] for s in specs if s['label']]
        labels=[s['label'] for s in specs if s['label']]
    else:
        ticks=[s['x'] for s in specs];labels=[s['label'] for s in specs]
    ax.set_xticks(ticks,labels,fontsize=8)
    for name,center in centers:
        ax.text(center,-.15,name,transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=10,fontweight='bold')
    ax.set_xlim(-.8,max(s['x'] for s in specs)+.8)
    ax.set_ylim(0,limit)
    ax.grid(axis='y',color='#e3e8ef');ax.set_axisbelow(True)
    if metric=='accuracy':
        ax.set_yticks([0,.25,.5,.75,1]);ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    elif metric=='reasoning_score':
        ax.set_yticks([0,1,2,3,4])
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    n={'all':6,'pool_shots':18,'pool_pattern':12,'pool_all':36}[view]
    ax.text(.5,-.29,f'H=5,20 pooled | responses/bar: 07=20; 08={n}',transform=ax.transAxes,ha='center',fontsize=8,color='#596579')
    if view in ('all','pool_shots'):
        ax.legend(handles=[Patch(facecolor='#dddddd',edgecolor='#555555',hatch=PATTERN_HATCHES[p],label=f'P{i+1}') for i,p in enumerate(('pattern1','pattern2'))],
                  loc='upper right',fontsize=8,ncol=2,frameon=False,bbox_to_anchor=(1,1.15))


def main():
    records,hashes=load_records()
    summary=[r for task in TASKS for view in VIEWS for r in summarize(records,task,view)]
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'hatch.linewidth':.8})
    for name,rows in [('per_query_metrics.csv',records),('group_metrics.csv',summary)]:
        with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for task in TASKS:
        titles={'accuracy':'Action accuracy (%)','mae':'MAE: action ID' if task==TASKS[0] else 'MAE: torque',
                'reasoning_score':'Mean reasoning level (new scale 1–4)'}
        limits={'accuracy':1.23,'reasoning_score':4.65,
                'mae':max(.1,max(r['mean'] for r in summary if r['task']==task and r['metric']=='mae' and r['mean'] is not None)*1.3)}
        fig,axes=plt.subplots(4,3,figsize=(24,17))
        fig.subplots_adjust(left=.045,right=.99,top=.89,bottom=.115,hspace=.9,wspace=.19)
        fig.suptitle(f'{task} | Existing 07 zero-shot vs 08 few-shot',fontsize=25,y=.98)
        fig.text(.5,.951,'07: n=10 per original condition (GPT-3.5 first 10)   |   08: n=3 per original condition   |   H=5 and 20 pooled',ha='center',fontsize=12)
        for i,view in enumerate(VIEWS):
            for j,metric in enumerate(METRICS):
                draw(axes[i,j],task,view,metric,summary,limits[metric])
                axes[i,j].set_title(titles[metric],fontsize=12,pad=30)
                # A standalone copy preserves the same metric scale.
                single,ax=plt.subplots(figsize=(12,5.2))
                single.subplots_adjust(left=.07,right=.985,top=.75,bottom=.26)
                draw(ax,task,view,metric,summary,limits[metric])
                single.suptitle(f'{task} | {VIEW_NAMES[view]}\n{titles[metric]}',fontsize=16,y=.98)
                single.text(.07,.015,'Different episodes/time points across 07 and 08; descriptive comparison, not paired causal effects.',fontsize=9,color='#596579')
                single.savefig(OUT/f'{task}_{view}_{metric}.png',dpi=180,facecolor='white');plt.close(single)
            y=axes[i,0].get_position().y1+.046
            fig.text(.045,y,VIEW_NAMES[view],fontsize=14,fontweight='bold')
        fig.text(.045,.022,'Different episodes/time points across 07 and 08. Repeated baseline bars are not additional samples. No independence-based error bars.\n'
                 '07 reasoning converted once: new = 5 − old. 08 scores unchanged. P1/P2 refer to the original few-shot selection patterns. MAE lower is better.',fontsize=11,color='#596579',linespacing=1.6)
        fig.savefig(OUT/f'{task}_overview.png',dpi=180,facecolor='white');plt.close(fig)
    audit=dict(input_sha256=hashes,selected07=160,selected08=144,
               missing_responses=sum(not r['response_available'] for r in records),
               missing_mae=sum(r['absolute_error'] is None for r in records),
               missing_reasoning=sum(r['reasoning_score'] is None for r in records),
               accuracy_parse_failures=sum(r['status']=='ignored' for r in records),
               group_rows=len(summary),png_files=26)
    (OUT/'validation.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    print(json.dumps({k:v for k,v in audit.items() if k!='input_sha256'},ensure_ascii=False,indent=2))
    print(OUT)


if __name__=='__main__':
    main()
