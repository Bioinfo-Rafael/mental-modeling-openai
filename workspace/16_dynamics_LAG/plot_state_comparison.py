"""Plot per-component paired state errors from the saved 15-vs-16 metrics."""
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'results/dynamics_batch_v1/analysis/comparison_with_15_v1'
OUTPUT=SOURCE/'state_components'
FIELDS=['lon_deg','lat_deg','alt_m','roll_deg','pitch_deg','yaw_deg','tas_mps','vn_mps','ve_mps','vu_mps']
UNITS=['deg','deg','m','deg','deg','deg','m/s','m/s','m/s','m/s']
MODELS=['sol','terra','luna']
SCOPES=['self','all_aircraft_missiles']


def main():
    if OUTPUT.exists():raise FileExistsError('Preserve existing plots; choose a new output path.')
    data=json.loads((SOURCE/'metrics.json').read_text())
    groups={(g['group']['model'],g['group']['scope']):g for g in data['groups'] if g['group']['target']=='all'}
    OUTPUT.mkdir()
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True})
    inventory=[]
    for field,unit in zip(FIELDS,UNITS):
        metric='state_sae.'+field
        fig,axs=plt.subplots(1,2,figsize=(13,5.5),sharey=True,layout='constrained')
        for ax,scope,title in zip(axs,SCOPES,['Self','All aircraft + missiles']):
            pos=np.arange(3);width=.34
            selected=[groups[model,scope]['metrics'][metric] for model in MODELS]
            for offset,key,color,label in [(-width/2,'exp15','#4779ac','15: original prompt'),(width/2,'exp16','#e45b22','16: dynamics prompt')]:
                values=[m[key] for m in selected]
                bars=ax.bar(pos+offset,values,width,color=color,label=label)
                ax.bar_label(bars,labels=[f'{v:.3g}' for v in values],padding=3,fontsize=10)
            ticks=[]
            for model,m in zip(MODELS,selected):
                p=m['reduction_pct']
                effect='n/a' if p is None else f'{abs(p):.1f}% '+('lower' if p>0 else 'higher' if p<0 else 'change')
                ticks.append(f'{model}\nn={m["n_pairs"]}/10\n{effect} in 16')
            ax.set_xticks(pos,ticks)
            ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.margins(y=.22)
            ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,4),useOffset=False)
        axs[0].set_ylabel(f'Mean sum of 10 absolute errors ({unit})')
        axs[0].legend(fontsize=9)
        fig.suptitle(f'{field} | Experiment 15 vs 16',fontsize=17)
        fig.supxlabel('Same valid response pairs; A0100 + A0200 pooled. Lower error is better.\nRoll/yaw use shortest-angle errors. Divide by 10 for per-step MAE.',fontsize=10)
        filename=f'{field}.png'
        fig.savefig(OUTPUT/filename,dpi=150);plt.close(fig)
        inventory.append({'field':field,'unit':unit,'image':filename,
                          'conditions':[{ 'model':model,'scope':scope,**groups[model,scope]['metrics'][metric]} for scope in SCOPES for model in MODELS]})
    columns=[(model,scope) for scope in SCOPES for model in MODELS]
    values=np.array([[groups[m,s]['metrics']['state_sae.'+f]['reduction_pct'] for m,s in columns] for f in FIELDS],dtype=float)
    limit=max(1,float(np.nanmax(np.abs(values))))
    fig,ax=plt.subplots(figsize=(12,8),layout='constrained')
    cmap=matplotlib.colors.LinearSegmentedColormap.from_list('red_white_blue', ['#b2182b', '#ffffff', '#2166ac'], N=257)
    color=ax.imshow(values,cmap=cmap,vmin=-limit,vmax=limit,aspect='auto')
    ax.set_yticks(np.arange(10),FIELDS)
    ax.set_xticks(np.arange(6),[f'{m}\n{"self" if s=="self" else "all aircraft"}\nn={groups[m,s]["n_common_valid"]}/10' for m,s in columns])
    ax.axvline(2.5,color='white',lw=3)
    for i in range(10):
        for j in range(6):
            v=values[i,j]
            ax.text(j,i,f'{v:+.1f}%',ha='center',va='center',fontsize=11,color='white' if abs(v)>.65*limit else '#172333')
    fig.colorbar(color,ax=ax,label='Error reduction in 16 (%) = 100 × (15 - 16) / 15')
    ax.set_title('State-component error reduction | Experiment 15 → 16',fontsize=17,pad=16)
    fig.supxlabel('Positive / blue = improvement; 0% / white = unchanged; negative / red = worse. Same valid pairs within each column.\nColumns can use different windows because malformed answers are excluded.',fontsize=10)
    fig.savefig(OUTPUT/'state_reduction_overview.png',dpi=150);plt.close(fig)
    (OUTPUT/'figure_index.json').write_text(json.dumps(inventory,indent=2)+'\n')
    cards='\n'.join(f'<section id="{item["field"]}"><h2>{item["field"]}</h2><a href="{item["image"]}"><img src="{item["image"]}" loading="lazy" alt="{item["field"]}の15と16の誤差比較"></a></section>' for item in inventory)
    nav=' / '.join(f'<a href="#{f}">{f}</a>' for f in FIELDS)
    page='''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>15と16：state成分別の比較</title><style>body{max-width:1300px;margin:30px auto;padding:0 20px;font:16px/1.7 system-ui;color:#24364a;background:#f5f7fa}section{background:white;padding:20px;margin:24px 0;border-radius:10px}img{width:100%;height:auto}a{color:#2459a6}h2{font-size:21px}nav{line-height:2.2}</style>
<h1>15と16：state全10成分の比較</h1>
<p>各モデル・入力範囲で、15と16の両方が採点可能な同じ回答だけを比較。A0100とA0200を集計しています。</p>
<p>成分別グラフの縦軸は<strong>10ステップの絶対誤差合計を回答間で平均</strong>した値です（小さいほど良い）。10で割ると1ステップ平均MAEになります。roll・yawは最短角度差です。</p>
<p>概観図は誤差削減率。正・青が改善、0%・白が変化なし、負・赤が悪化です。形式失敗により列ごとの採用問題が異なるため、モデル間比較の母集団は必ずしも一致しません。</p>
<p>全体の形式成功数は15が53/60、16が48/60、共通有効回答は42件です。16の回帰係数には評価対象試合が含まれ、未知試合への汎化評価ではありません。</p>
<nav>__NAV__</nav><section><h2>誤差削減率の概観</h2><a href="state_reduction_overview.png"><img src="state_reduction_overview.png" alt="全10成分の誤差削減率"></a></section>__CARDS__
<p><a href="../metrics.csv">数値一覧CSV</a> / <a href="../summary.txt">比較の要約</a></p></html>'''
    (OUTPUT/'index.html').write_text(page.replace('__NAV__',nav).replace('__CARDS__',cards))
    assert all((OUTPUT/item['image']).is_file() for item in inventory)
    print('Saved 10 component plots, reduction overview, and gallery:',OUTPUT/'index.html')


if __name__=='__main__':main()
