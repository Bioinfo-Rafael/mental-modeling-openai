"""Plot recorded target-aircraft time series around each actual saved question.

30 question groups x 5 H x 4 tasks = 600 condition views. No API calls.
"""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from build_results_viewer import HERE, ex
from lag_data.dataset import load_episode, index_records, PROCESSED
from lag_data.definitions import STATE_FEATURES, STATE_UNITS, ACTIONS, ACTION_LABELS
from lag_experiment.runner import build_query
from lag_experiment.prompts import build_messages
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HISTORY_COLOR = '#DDECFB'
CONTEXT_COLOR = '#BA6B14'
TARGET_COLOR = '#C5334D'
SERIES_COLOR = '#376A93'


def numeric(value):
    return None if pd.isna(value) else float(value)


def recorded_window(records, target, t, before=50, after=10):
    """No interpolation: absent, invalid, or dead-object values stay missing."""
    window = []
    for step in range(t-before, t+after+1):
        row = records.get((step, target))
        alive = row is not None and pd.notna(row['alive']) and row['alive'] == 1
        state_ok = alive and pd.notna(row['state_valid']) and row['state_valid'] == 1
        action_ok = alive and pd.notna(row['action_status']) and row['action_status'] == 'recorded'
        states = [numeric(row[f]) for f in STATE_FEATURES] if state_ok else [None]*10
        actions = [numeric(row[f'{f}_id']) for f in ACTIONS] if action_ok else [None]*4
        window.append(dict(step=step, relative_step=step-t, state=states, action=actions))
    return window


def line_values(values, circular=False):
    """Break the line across angular wraps; plotted points remain unmodified."""
    output = np.array([np.nan if x is None else x for x in values], float)
    if circular:
        jumps = np.abs(np.diff(output)) > 180
        output[np.flatnonzero(jumps)+1] = np.nan
    return output


def visible_steps(visible):
    result = dict(state=[], action=[])
    target = visible['target_id']
    for kind in result:
        result[kind] = sorted({block['step'] for block in visible['query_context']
                               if any(r['object_id'] == target for r in block.get(kind+'s', []))})
    return result


def draw(sample, visible, window, output, before, after):
    t, h = sample['target_step'], sample['H']
    context = visible_steps(visible)
    x = np.array([r['relative_step'] for r in window])
    fig, axes = plt.subplots(7, 2, figsize=(15, 16), sharex=True)
    fig.subplots_adjust(left=.085, right=.98, bottom=.09, top=.865, hspace=.50, wspace=.28)
    title = f"{sample['episode_id']} | {sample['target_id']} | question {sample['sample_number']+1}/5"
    fig.suptitle(title, fontsize=16, y=.986)
    fig.text(.5,.960,f"{sample['task']} | H={h} | target t={t} ({t*.2:.1f} s) | recorded window [{t-before}, {t+after}]",ha='center',fontsize=12)
    fig.legend(handles=[Line2D([],[],color=SERIES_COLOR,label='Recorded truth (includes unseen context)'),
                        Patch(facecolor=HISTORY_COLOR,label='History supplied to model'),
                        Line2D([],[],marker='D',linestyle='',color=CONTEXT_COLOR,label='Additional supplied values'),
                        Line2D([],[],marker='*',linestyle='',markersize=11,color=TARGET_COLOR,label='Requested target (hidden from model)')],
               loc='upper center',bbox_to_anchor=(.5,.946),ncol=2,frameon=False,fontsize=10)
    # Paired state columns first, action columns last: 10 states + 4 actions.
    panels = [('state',i,f) for i,f in enumerate(STATE_FEATURES)] + [('action',i,f) for i,f in enumerate(ACTIONS)]
    for ax,(kind,i,field) in zip(axes.flat,panels):
        values = [r[kind][i] for r in window]
        y = np.array([np.nan if v is None else v for v in values],float)
        ax.axvspan(sample['history_steps'][0]-t-.5,sample['history_steps'][-1]-t+.5,color=HISTORY_COLOR,zorder=0)
        ax.axvline(0,color=TARGET_COLOR,linestyle='--',linewidth=1,alpha=.75)
        if kind == 'state':
            ax.plot(x,line_values(values,field in ('roll_deg','yaw_deg')),color=SERIES_COLOR,lw=1.3)
            ax.scatter(x,y,s=7,color=SERIES_COLOR,zorder=2)
            ax.set_ylabel(STATE_UNITS[field],fontsize=9)
            ax.ticklabel_format(axis='y',style='plain',useOffset=False)
        else:
            ax.step(x,y,where='post',color=SERIES_COLOR,lw=1.4)
            ax.scatter(x,y,s=7,color=SERIES_COLOR,zorder=2)
            labels=ACTION_LABELS[field]
            ax.set_yticks(range(len(labels)),[f'{n}: {label}' for n,label in enumerate(labels)],fontsize=8)
            ax.set_ylim(-.3,len(labels)-.7)
        for step in context[kind]:
            relative=step-t
            if -before <= relative <= after:
                ax.scatter([relative],[y[relative+before]],marker='D',s=48,color=CONTEXT_COLOR,edgecolors='white',linewidths=.5,zorder=5)
        requested = field in sample['prediction_fields']
        if requested:
            ax.scatter([0],[y[before]],marker='*',s=125,color=TARGET_COLOR,edgecolors='white',linewidths=.4,zorder=6)
        ax.set_title(field+('  [predict at t]' if requested else ''),loc='left',fontsize=10)
        ax.grid(alpha=.20)
        ax.spines[['top','right']].set_visible(False)
        ax.set_xlim(-before,after)
        ax.set_xticks(sorted(set([-before,-40,-30,-20,-10,0,after]) & set(range(-before,after+1))))
        ax.tick_params(axis='x',labelbottom=True,labelsize=8)
    for ax in axes[-1]:ax.set_xlabel('Step relative to target t (1 step = 0.2 s)',fontsize=10)
    fig.text(.5,.022,'Target-aircraft telemetry; identical target series for self / all scopes and all 3 models.\n'
             'The full curve is retrospective context, not all supplied to the model. Blank = absent / invalid / dead.\n'
             'Shoot is supplied in known actions but is not an action-prediction target. Roll/yaw lines break at angular wraps.',
             ha='center',fontsize=9,color='#4B5563',linespacing=1.6)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=115,facecolor='white')
    plt.close(fig)


def gallery(output, items, before, after):
    payload=json.dumps(items,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    page='''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LAG 問題周辺の時系列</title><style>body{font:15px/1.7 system-ui;color:#192738;background:#f4f7fb;max-width:1700px;margin:auto;padding:24px}nav,section{background:white;border:1px solid #d5deeb;border-radius:10px;padding:18px;margin:16px 0}nav{position:sticky;top:0;z-index:1;display:flex;gap:14px;flex-wrap:wrap;align-items:center}select,button{font:inherit;padding:7px;border:1px solid #bcc9dc;border-radius:5px;background:white}label{display:flex;flex-direction:column}img{width:100%;height:auto}a{color:#2457cf}.blue{background:#ddecfb}.orange{color:#a85c0e}.red{color:#c5334d}#status{font-weight:600}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid #dae0eb;padding:6px;text-align:left}.scroll{overflow:auto}</style><h1>Question周辺の状態・行動の時系列</h1><p>5問×3試合×2機体＝30グループ。H・課題で回答対象tが変わるため、各グループ内で条件を選びます。全600条件、正解時刻は177通りです。</p><p>横軸は回答対象tからのstep差（1step＝0.2秒）。状態10成分と行動4成分の実測値を表示します。<span class="blue">青背景＝プロンプトの履歴</span>、<span class="orange">橙の菱形＝追加で与えた値</span>、<span class="red">赤い星＝回答対象の正解</span>です。</p><p>青い曲線には、モデルへ見せていない前後の実測値も含みます。対象機体の系列は3モデル・self／全機体で共通です。全機体条件で追加された他機体・ミサイルの系列はこの図の対象外です。試合開始前・欠損・無効な状態は空白です。</p><nav><label>試合<select id="episode"></select></label><label>機体<select id="target"></select></label><label>問題<select id="sample"></select></label><label>H<select id="history"></select></label><label>課題<select id="task"></select></label><button id="previous">前の問題</button><button id="next">次の問題</button></nav><section><p id="status"></p><p id="visible"></p><a id="download">PNGを開く</a> · <a href="timeline_index.json">条件と時刻の一覧JSON</a><details><summary>このグループのH・課題ごとの実際のt</summary><div class="scroll"><table id="times"></table></div></details></section><img id="plot" alt="選択した問題周辺の状態と行動の時系列"><script id="data" type="application/json">__DATA__</script><script>
const rows=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);
function options(id,values,labels){values.forEach((v,i)=>$(id).append(new Option(labels?labels[i]:v,v)))}
options('episode',[...new Set(rows.map(r=>r.episode_id))]);options('target',['A0100','A0200']);options('sample',[0,1,2,3,4],['1 / 5','2 / 5','3 / 5','4 / 5','5 / 5']);options('history',[2,5,10,20,30]);options('task',['next-action','last-action','next-state','last-state']);
function group(){return rows.filter(r=>r.episode_id===$('episode').value&&r.target_id===$('target').value&&r.sample_number===Number($('sample').value))}
function update(){const g=group(),r=g.find(r=>r.H===Number($('history').value)&&r.task===$('task').value);$('plot').src=r.png;$('download').href=r.png;$('status').textContent=`${r.episode_id} / ${r.target_id} / 問${r.sample_number+1} / H=${r.H} / ${r.task} / t=${r.target_step} (${(r.target_step*.2).toFixed(1)}秒)`;$('visible').textContent=`表示区間: step ${r.window_start}〜${r.window_end}。履歴: ${r.history_steps[0]}〜${r.history_steps.at(-1)}。追加の状態: ${r.context_steps.state.join(', ')||'なし'}。追加の行動: ${r.context_steps.action.join(', ')||'なし'}。`;$('previous').disabled=r.sample_number===0;$('next').disabled=r.sample_number===4;const t=$('times');t.replaceChildren();const tr=document.createElement('tr');['H','next-action','last-action','next-state','last-state'].forEach(v=>{const e=document.createElement('th');e.textContent=v;tr.append(e)});t.append(tr);[2,5,10,20,30].forEach(h=>{const tr=document.createElement('tr');[h,...['next-action','last-action','next-state','last-state'].map(task=>g.find(x=>x.H===h&&x.task===task).target_step)].forEach(v=>{const e=document.createElement('td');e.textContent=v;tr.append(e)});t.append(tr)})}
document.querySelectorAll('select').forEach(e=>e.onchange=update);$('previous').onclick=()=>{$('sample').value=Number($('sample').value)-1;update()};$('next').onclick=()=>{$('sample').value=Number($('sample').value)+1;update()};update();
</script></html>'''
    (output/'index.html').write_text(page.replace('__DATA__',payload),encoding='utf-8')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='sliding_window_v1')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--before',type=int,default=50)
    parser.add_argument('--after',type=int,default=10)
    args=parser.parse_args(argv)
    if args.before<30 or args.after<1:parser.error('Need --before >= 30 and --after >= 1 to show all supplied values')
    source=(HERE/'results'/args.run).resolve()
    if not source.is_relative_to((HERE/'results').resolve()):parser.error('--run must be inside results')
    output=args.output_dir or source/'analysis'/('question_timelines_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    if output.exists():parser.error('Choose a new output directory; existing figures are not overwritten')
    manifest=source/'manifest.json';plan=json.loads(manifest.read_text());ex.verify_plan(plan)
    samples=[(sid,s) for sid,s in plan['samples'].items() if s['observation_scope']=='self']
    samples.sort(key=lambda pair:tuple(pair[1][k] for k in ('episode_id','target_id','sample_number','H','task')))
    if len(samples)!=600:raise ValueError('Expected 600 distinct self conditions')
    pairs=defaultdict(dict)
    for sid,s in plan['samples'].items():
        key=tuple(s[k] for k in ('episode_id','target_id','sample_number','H','task'))
        pairs[key][s['observation_scope']]=s
    for group in pairs.values():
        if set(group)!={'self','all_aircraft_missiles'}:raise ValueError('Missing scope pair')
        for k in ('ground_truth','start','target_step','history_steps'):
            if group['self'][k]!=group['all_aircraft_missiles'][k]:raise ValueError('Scopes have different target windows')
    episode_records={};hashes={}
    for episode in sorted({s['episode_id'] for _,s in samples}):
        frame=load_episode(PROCESSED/f'{episode}.csv')
        if any(s['csv_sha256']!=frame.attrs['csv_sha256'] for _,s in samples if s['episode_id']==episode):raise ValueError('CSV differs from experiment plan')
        hashes[episode]=frame.attrs['csv_sha256'];episode_records[episode]=index_records(frame)
    prepared=[]
    # Reconstruct every supplied prompt and ground truth BEFORE writing outputs.
    for sid,s in samples:
        records=episode_records[s['episode_id']]
        visible,truth,q=build_query(records,task=s['task'],history_length=s['H'],start=s['start'],target_id=s['target_id'])
        if truth!=s['ground_truth'] or q!=s['query_step'] or build_messages(visible)!=s['messages']:
            raise ValueError('Reconstructed question differs from saved prompt')
        window=recorded_window(records,s['target_id'],s['target_step'],args.before,args.after)
        key='action' if s['task'].endswith('action') else 'state'
        if window[args.before][key][:len(truth)]!=truth:raise ValueError('Timeline target differs from saved ground truth')
        prepared.append((sid,s,visible,window))
    output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.unicode_minus':False})
    inventory=[]
    for i,(sid,s,visible,window) in enumerate(prepared,1):
        relative=Path(s['episode_id'])/s['target_id']/f"question_{s['sample_number']+1:02d}"/f"H{s['H']:02d}_{s['task']}.png"
        draw(s,visible,window,output/relative,args.before,args.after)
        inventory.append(dict(sample_id=sid,**{k:s[k] for k in ('episode_id','target_id','sample_number','H','task','target_step','query_step','start','history_steps','prediction_fields','ground_truth')},
                              png=str(relative),context_steps=visible_steps(visible),window_start=s['target_step']-args.before,window_end=s['target_step']+args.after))
        if i%20==0:print(f'{i}/600 views ({i//20}/30 question groups)',flush=True)
    metadata=dict(plan_id=plan['plan_id'],manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),csv_sha256=hashes,
                  views=len(inventory),groups=len({(s['episode_id'],s['target_id'],s['sample_number']) for _,s in samples}),
                  unique_target_times=len({(s['episode_id'],s['target_id'],s['target_step']) for _,s in samples}),
                  dt_s=.2,before=args.before,after=args.after,prompts_reconstructed_and_verified=600,
                  notes=['One model-independent target-aircraft timeline per self/all paired problem.',
                         'All 10 states and 4 action IDs, including shoot (not predicted).',
                         'Recorded context outside supplied history is retrospective and not shown to the model.',
                         'Missing, invalid, or dead-object values are gaps, never interpolated. Angular wrap lines broken without modifying point values.'])
    for name,data in [('timeline_index.json',inventory),('metadata.json',metadata)]:
        (output/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    gallery(output,inventory,args.before,args.after)
    print(f'Saved {len(inventory)} PNG views / 30 groups: {output/"index.html"}',flush=True)


if __name__=='__main__':main()
