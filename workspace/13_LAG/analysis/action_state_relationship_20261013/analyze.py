"""Observed action/state associations and held-out incremental prediction value.

Reads only the three named CSVs. No source memo or simulator code is read.
"""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'velocity_position_fit_20261013'))
from fit import RAW,FILES
FIELDS=['lon_deg','lat_deg','alt_m','roll_deg','pitch_deg','yaw_deg','tas_mps','vn_mps','ve_mps','vu_mps']
ACTS={'altitude_action':['down','hold','up'],
      'heading_action':['left30','left15','hold','right15','right30'],
      'speed_action':['down','hold','up']}
ONEHOT=[(field,label) for field,labels in ACTS.items() for label in labels if label!='hold']
ALPHAS=[0.001,.1,1.,10.,100.,1000.]


def wrap(x):return (x+180)%360-180


def valid(rows,start,end):
    ww=rows[start:end+1]
    return all(r['alive']=='1' for r in ww) and all(int(ww[i+1]['step'])-int(ww[i]['step'])==1 and
           abs(float(ww[i+1]['time_s'])-float(ww[i]['time_s'])-.2)<1e-9 for i in range(len(ww)-1))


def read_data():
    features=[];actions=[];targets=[];groups=[];observations=[];sources=[]
    for filename in FILES:
        path=RAW/filename
        with path.open() as f: rows=list(csv.DictReader(f))
        assert len({(r['uid'],r['step']) for r in rows})==len(rows)
        sources.append({'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        for uid in ['A0100','A0200']:
            rr=sorted((r for r in rows if r['uid']==uid),key=lambda r:int(r['step']))
            states=np.array([[float(r[k]) for k in FIELDS] for r in rr])
            for n,row in enumerate(rr):
                if row['alive']!='1' or not all(row[a] in ACTS[a] for a in ACTS):continue
                record={'match':filename,'uid':uid,'step':int(row['step']),
                        **{a:row[a] for a in ACTS},'current_vu':states[n,9],
                        'current_roll':states[n,3],'current_tas':states[n,6]}
                for h in [1,5,10]:
                    if n+h<len(rr) and valid(rr,n,n+h):
                        record[f'dalt_{h}']=states[n+h,2]-states[n,2]
                        record[f'dyaw_{h}']=wrap(states[n+h,5]-states[n,5])
                        record[f'dtas_{h}']=states[n+h,6]-states[n,6]
                        # Later actions are observed only to describe persistence,
                        # never supplied to the predictive model.
                        for a in ACTS:record[f'{a}_unchanged_{h}']=all(rr[k][a]==row[a] for k in range(n,n+h))
                observations.append(record)
                if n<2 or n+1>=len(rr) or not valid(rr,n-2,n+1):continue
                x=[]
                for k in [n,n-1,n-2]:
                    s=states[k];angles=np.deg2rad(s[3:6])
                    x.extend(np.r_[s[[0,1,2,6,7,8,9]],np.sin(angles),np.cos(angles)])
                x.extend(wrap(states[n,3:6]-states[n-1,3:6]))
                x.extend(wrap(states[n-1,3:6]-states[n-2,3:6]))
                y=states[n+1]-states[n];y[[3,5]]=wrap(y[[3,5]])
                features.append(x);targets.append(y);actions.append([int(row[f]==label) for f,label in ONEHOT])
                groups.append(filename)
    return np.array(features),np.array(actions),np.array(targets),np.array(groups),observations,sources


def fit(x,y,alphas):
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-12]=1
    z=(x-mean)/scale;ym=y.mean(0)
    u,s,vt=np.linalg.svd(z,full_matrices=False)
    beta=np.column_stack([(vt.T*(s/(s*s+a)))@(u.T@(y[:,j]-ym[j])) for j,a in enumerate(alphas)])
    raw=beta/scale[:,None]
    return raw,ym-mean@raw


def error(pred,y):
    e=pred-y;e[:,[3,5]]=wrap(e[:,[3,5]])
    return e


def tune(x,y,group):
    losses=np.zeros((len(ALPHAS),len(FIELDS)))
    for i,a in enumerate(ALPHAS):
        for heldout in np.unique(group):
            test=group==heldout;b,c=fit(x[~test],y[~test],[a]*len(FIELDS))
            losses[i]+=np.sum(error(x[test]@b+c,y[test])**2,axis=0)
    return np.array(ALPHAS)[np.argmin(losses,axis=0)]


def prediction(x,onehot,y,group):
    output={};preds={};folds=[]
    for name,z in [('state_history',x),('state_history_and_action',np.column_stack((x,onehot)))]:
        p=np.empty_like(y)
        for match in FILES:
            test=group==match
            aa=tune(z[~test],y[~test],group[~test]);b,c=fit(z[~test],y[~test],aa)
            p[test]=z[test]@b+c
            folds.append({'test_match':match,'model':name,'windows':int(test.sum()),'alphas':aa.tolist(),
                          'rmse':np.sqrt(np.mean(error(p[test],y[test])**2,axis=0)).tolist()})
        e=error(p,y);preds[name]=p
        output[name]={'rmse':np.sqrt(np.mean(e*e,axis=0)).tolist(),'mae':np.mean(np.abs(e),axis=0).tolist()}
    a=np.array(output['state_history']['rmse']);b=np.array(output['state_history_and_action']['rmse'])
    return {'n_windows':len(y),'target_fields':FIELDS,'scores':output,
            'rmse_reduction_pct':(100*(a-b)/a).tolist(),'folds':folds}


def grouped(observations):
    result=[]
    for a,labels in ACTS.items():
        for label in labels:
            rows=[r for r in observations if r[a]==label]
            out={'action':a,'label':label,'n_current':len(rows),'metrics':{}}
            for metric in ['current_vu','current_roll','current_tas']+[f'{k}_{h}' for k in ['dalt','dyaw','dtas'] for h in [1,5,10]]:
                vals=np.array([r[metric] for r in rows if metric in r])
                out['metrics'][metric]={'n':len(vals),'mean':float(vals.mean()) if len(vals) else None,
                                        'median':float(np.median(vals)) if len(vals) else None}
            for h in [5,10]:
                key=f'{a}_unchanged_{h}';vv=[r[key] for r in rows if key in r]
                out[f'action_unchanged_through_{h}_steps_fraction']=float(np.mean(vv)) if vv else None
            out['by_match']={f:sum(r['match']==f for r in rows) for f in FILES}
            result.append(out)
    return result


def plot(observations):
    fig,axs=plt.subplots(3,3,figsize=(16,12),layout='constrained')
    specs=[('altitude_action','current_vu','dalt','Current vertical velocity (m/s)','Altitude change (m)'),
           ('heading_action','current_roll','dyaw','Current roll (deg)','Yaw change (deg)'),
           ('speed_action','current_tas','dtas','Current TAS (m/s)','TAS change (m/s)')]
    for i,(action,current,delta,current_title,delta_title) in enumerate(specs):
        for j,(metric,title) in enumerate([(current,current_title),(delta+'_1',delta_title+' over 0.2 s'),(delta+'_5',delta_title+' over 1.0 s')]):
            ax=axs[i,j];arrays=[np.array([r[metric] for r in observations if r[action]==a and metric in r]) for a in ACTS[action]]
            ax.boxplot(arrays,tick_labels=[f'{a}\nn={len(v)}' for a,v in zip(ACTS[action],arrays)],showfliers=False,showmeans=True,
                       meanprops={'marker':'D','markerfacecolor':'#e45b22','markeredgecolor':'#e45b22','markersize':4})
            ax.axhline(0,color='#777777',lw=.7);ax.set_title(title,loc='left');ax.set_xlabel(action)
            ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Observed action / state association | A0100 + A0200, three matches',fontsize=17)
    fig.supxlabel('Boxes: median and interquartile range; whiskers: 1.5 IQR; orange diamonds: mean. Outliers hidden only in this plot.\nNo causal claim: state influences action choice; later actions may change during the 1-second interval.',fontsize=11)
    fig.savefig(HERE/'action_state_distributions.png',dpi=140);plt.close(fig)


def main():
    x,a,y,g,obs,sources=read_data()
    assert np.isfinite(x).all() and np.isfinite(y).all()
    results={'sources':sources,'current_action_rows':len(obs),'action_summary':grouped(obs),
             'prediction':prediction(x,a,y,g),
             'method':{'state_features':'45 features: for n,n-1,n-2, lon/lat/alt/TAS/vN/vE/vU and sin/cos of roll/pitch/yaw (39); two wrapped backward angle increments (6).',
                'action_features':[f'{f}={a}' for f,a in ONEHOT],
                'prediction_target':'One-step increments of all ten state fields in their original units; roll/yaw wrapped.',
                'validation':'Outer leave-one-match-out; independent inner leave-one-training-match-out ridge tuning per output and per model. Only training data used for scaling.',
                'scope':'Only alive A0100/A0200 rows with recorded movement actions. Blue aircraft have no recorded movement actions. Shoot is excluded from this aircraft-motion analysis.',
                'causality':'Observational, temporally dependent samples. Action is selected from state; subsequent actions may change. No randomized intervention or significance claim.',
                'limitation':'Three matches, two share the same initial configuration; pooled associations can reflect policy/match mixture. Ridge is only one predictive specification.'}}
    (HERE/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    plot(obs)
    p=results['prediction'];lines=['Action/state association analysis, 2026-10-13 JST',
        f'Current action rows: {len(obs)}; predictive windows: {len(y)}',
        'Held-out one-step RMSE: state history -> state history + current movement action']
    for i,f in enumerate(FIELDS):lines.append(f'{f}: {p["scores"]["state_history"]["rmse"][i]:.8g} -> {p["scores"]["state_history_and_action"]["rmse"][i]:.8g}; reduction {p["rmse_reduction_pct"][i]:+.2f}%')
    lines+=['','Action-conditioned mean changes (descriptive, not causal):']
    for row in results['action_summary']:
        metric={'altitude_action':'dalt','heading_action':'dyaw','speed_action':'dtas'}[row['action']]
        lines.append(f'{row["action"]} {row["label"]}: '+json.dumps({f'{h} steps':row['metrics'][f'{metric}_{h}'] for h in [1,5,10]}))
    lines+=['',json.dumps(results['method'],indent=2)]
    (HERE/'summary.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:25]))


if __name__=='__main__':main()
