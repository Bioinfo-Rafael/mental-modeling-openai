"""Refresh result tables and plot per-task few-shot accuracy (no API calls)."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
if __package__:
    from .summarize_results import main as summarize
else:
    from summarize_results import main as summarize

HERE = Path(__file__).resolve().parents[1]
COLORS = {4: '#2468a2', 8: '#65a9ca', 12: '#db9459'}


def main():
    summarize()
    data = json.loads((HERE/'results/analysis/accuracy_tables.json').read_text())
    rows = data['rows']
    lookup = {(r['task'], r['H'], r['score_pattern'], r['shots'], r['model']):r for r in rows}
    output = HERE/'results/analysis/figures'
    output.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                         'axes.spines.top':False,'axes.spines.right':False})
    for task in dict.fromkeys(r['task'] for r in rows):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
        fig.subplots_adjust(left=.085, right=.98, top=.79, bottom=.18, hspace=.42, wspace=.17)
        fig.suptitle(f'{task} | Next-action accuracy', fontsize=21, y=.975)
        fig.text(.5,.925,'Pattern 1: Correct S1 / Incorrect S3–S4    •    Pattern 2: Correct S1–S2 / Incorrect S1–S3',
                 ha='center',fontsize=11)
        fig.legend(handles=[Patch(facecolor=COLORS[s],label=f'2k = {s} ({s//2} correct + {s//2} incorrect)') for s in COLORS],
                   loc='upper center',bbox_to_anchor=(.5,.897),ncol=3,frameon=False,fontsize=10)
        for i, pattern in enumerate(('pattern1','pattern2')):
            for j,H in enumerate((5,20)):
                ax=axes[i,j]
                for model_index,model in enumerate(('terra','luna')):
                    for shot_index,shots in enumerate(COLORS):
                        r=lookup[(task,H,pattern,shots,model)]
                        x=model_index+(shot_index-1)*.23
                        value=r['accuracy']
                        if value is None:
                            ax.text(x,3,'N/A',ha='center',fontsize=10)
                            continue
                        height=value*100
                        ax.bar(x,height,width=.205,color=COLORS[shots],zorder=3)
                        if height==0:
                            ax.plot([x-.10,x+.10],[0,0],color=COLORS[shots],lw=3,zorder=4,clip_on=False)
                        label=f'{height:.0f}%' if height in (0,100) else f'{height:.1f}%'
                        label+=f'\n({r["correct"]}/{r["received"]})'
                        if r['received']!=r['expected']:
                            label+='*'
                        ax.text(x,height+2,label,ha='center',va='bottom',fontsize=9,linespacing=1.25)
                ax.set_title(f'Pattern {i+1}  |  H = {H}',fontsize=13,pad=12)
                ax.set_xlim(-.55,1.55)
                ax.set_ylim(0,119)
                ax.set_yticks([0,25,50,75,100])
                ax.yaxis.set_major_formatter(PercentFormatter(100,decimals=0))
                ax.set_xticks([0,1],['Terra','Luna'],fontsize=12)
                ax.grid(axis='y',color='#e5e9ef',zorder=0)
                if j==0:
                    ax.set_ylabel('Accuracy')
        task_rows=[r for r in rows if r['task']==task]
        received=sum(r['received'] for r in task_rows)
        expected=sum(r['expected'] for r in task_rows)
        ignored=sum(r['ignored'] for r in task_rows)
        fig.text(.085,.105,f'n = 3 per condition; labels show accuracy and correct / received.  Responses: {received}/{expected}.',fontsize=10)
        fig.text(.085,.079,f'Unparseable responses ({ignored} in this task) count as non-correct. Missing responses are excluded (* = incomplete).',fontsize=10)
        fig.text(.085,.047,f'Snapshot: {data["created_at_jst"]}   |   Same three target queries across models, shot counts and score patterns.',
                 fontsize=9,color='#596579')
        for extension in ('png',):
            path=output/f'{task}_accuracy.{extension}'
            fig.savefig(path,dpi=180,facecolor='white')
            print(path)
        plt.close(fig)


if __name__=='__main__':
    main()
