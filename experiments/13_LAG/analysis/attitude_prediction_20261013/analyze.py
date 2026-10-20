"""Causal one-step attitude prediction and nose/motion direction comparison.

Reads only the three named action CSVs and the previously authored fit.py.
Run with Python, numpy and matplotlib. No simulator/source memo is consulted.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'velocity_position_fit_20261013'))
from fit import RAW, FILES, DT, coordinates

ANGLES = ['Roll', 'Pitch', 'Yaw']
UIDS = ['A0100','A0200','B0100','B0200']
ALPHAS = np.array([0., .001, .1, 1., 10., 100., 1000.])
FEATURES = ([f'previous_increment_{a}_deg' for a in ANGLES]
            +[f'earlier_increment_{a}_deg' for a in ANGLES]
            +[f'sin_{a}' for a in ANGLES]+[f'cos_{a}' for a in ANGLES]
            +[f'velocity_{d}_mps' for d in ['N','E','U']]
            +[f'velocity_last_difference_{d}_mps' for d in ['N','E','U']]
            +[f'velocity_earlier_difference_{d}_mps' for d in ['N','E','U']]
            +['tas_mps','tas_last_difference_mps','tas_earlier_difference_mps'])


def wrap(a):
    return (a+180)%360-180


def direction(v):
    return np.column_stack((np.rad2deg(np.arctan2(v[:,1],v[:,0])),
                             np.rad2deg(np.arctan2(v[:,2],np.hypot(v[:,0],v[:,1])))))


def separation(a,b):
    dot=np.sum(a*b,axis=1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1))
    return np.rad2deg(np.arccos(np.clip(dot,-1,1)))


def dataset():
    items=[]; sources=[]
    for filename in FILES:
        with (RAW/filename).open() as f:
            rows=list(csv.DictReader(f))
        assert len({(r['uid'],r['step']) for r in rows})==len(rows)
        sources.append({'file':str(RAW/filename),'rows':len(rows),
                        'sha256':hashlib.sha256((RAW/filename).read_bytes()).hexdigest()})
        for uid in UIDS:
            rr=sorted((r for r in rows if r['uid']==uid),key=lambda r:int(r['step']))
            a=np.array([[float(r[k]) for k in ['roll_deg','pitch_deg','yaw_deg']] for r in rr])
            unwrapped=np.rad2deg(np.unwrap(np.deg2rad(a),axis=0))
            v=np.array([[float(r[k]) for k in ['vn_mps','ve_mps','vu_mps']] for r in rr])
            tas=np.array([float(r['tas_mps']) for r in rr])
            pos,ecef_v=coordinates(rr,'wgs84')
            for n in range(2,len(rr)-1):
                w=rr[n-2:n+2]
                if not all(r['alive']=='1' for r in w): continue
                if not np.all(np.diff([int(r['step']) for r in w])==1): continue
                if not np.allclose(np.diff([float(r['time_s']) for r in w]),DT,rtol=0,atol=1e-9): continue
                d1,d2=wrap(a[n]-a[n-1]),wrap(a[n-1]-a[n-2])
                x=np.r_[d1,d2,np.sin(np.deg2rad(a[n])),np.cos(np.deg2rad(a[n])),
                        v[n],v[n]-v[n-1],v[n-1]-v[n-2],tas[n],tas[n]-tas[n-1],tas[n-1]-tas[n-2]]
                lat,lon=np.deg2rad([float(rr[n]['lat_deg']),float(rr[n]['lon_deg'])])
                rot=np.array([[-np.sin(lat)*np.cos(lon),-np.sin(lat)*np.sin(lon),np.cos(lat)],
                              [-np.sin(lon),np.cos(lon),0],
                              [np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)]])
                displacement=rot@(pos[n+1]-pos[n])
                midpoint_v=rot@((ecef_v[n]+ecef_v[n+1])/2)
                theta,psi=np.deg2rad(a[n,1:])
                nose=np.array([np.cos(theta)*np.cos(psi),np.cos(theta)*np.sin(psi),np.sin(theta)])
                items.append({'match':filename,'uid':uid,'time':float(rr[n]['time_s']),
                              'x':x,'y':wrap(a[n+1]-a[n]),'current':unwrapped[n],
                              'next':unwrapped[n+1],'velocity':v[n], 'next_velocity':v[n+1],
                              'nose':nose,'displacement':displacement,'midpoint_velocity':midpoint_v})
    keys=items[0].keys()
    data={k:np.array([i[k] for i in items]) for k in keys}
    assert np.isfinite(data['x']).all() and np.isfinite(data['y']).all()
    assert np.allclose(wrap(data['next']-data['current']),data['y'])
    return data,sources


def train(x,y,alpha):
    mean=x.mean(axis=0); scale=x.std(axis=0); scale[scale<1e-12]=1
    z=(x-mean)/scale
    ymean=y.mean(axis=0)
    u,s,vt=np.linalg.svd(z,full_matrices=False)
    factors=np.divide(s,s*s+alpha,out=np.zeros_like(s),where=s*s+alpha>1e-20)
    beta=(vt.T*factors)@(u.T@(y-ymean))
    raw=beta/scale[:,None]
    return raw,ymean-mean@raw


def predict(x,model):
    return x@model[0]+model[1]


def tune(x,y,groups):
    unique=np.unique(groups)
    losses=np.zeros((len(ALPHAS),3))
    for ai,alpha in enumerate(ALPHAS):
        for group in unique:
            val=groups==group
            model=train(x[~val],y[~val],alpha)
            losses[ai]+=np.sum(wrap(predict(x[val],model)-y[val])**2,axis=0)
    return ALPHAS[np.argmin(losses,axis=0)]


def tuned_train(x,y,alphas):
    models=[train(x,y[:,j:j+1],alpha) for j,alpha in enumerate(alphas)]
    return np.column_stack([m[0][:,0] for m in models]),np.array([m[1][0] for m in models])


def error_stats(error):
    return {'rmse_deg':np.sqrt(np.mean(error**2,axis=0)).tolist(),
            'mae_deg':np.mean(np.abs(error),axis=0).tolist(),
            'p95_abs_deg':np.quantile(np.abs(error),.95,axis=0).tolist(),
            'max_abs_deg':np.max(np.abs(error),axis=0).tolist()}


def scalar_stats(x):
    return {'mean':float(np.mean(x)),'median':float(np.median(x)),
            'rmse':float(np.sqrt(np.mean(x*x))),
            'median_abs':float(np.median(np.abs(x))),'p95_abs':float(np.quantile(np.abs(x),.95)),
            'max_abs':float(np.max(np.abs(x)))}


def prediction_analysis(d):
    x,y,g=d['x'],d['y'],d['match']
    predictions={'persistence':np.zeros_like(y),'constant_rate':x[:,:3],
                 'constant_acceleration':2*x[:,:3]-x[:,3:6]}
    results={'angle_order':ANGLES,'features':FEATURES,'ridge_alpha_grid':ALPHAS.tolist(),
             'models':{},'final_models':{},'folds':[]}
    for name,count in [('history_only',12),('history_and_velocity',24)]:
        xx=x[:,:count]
        oof=np.zeros_like(y)
        for group in FILES:
            test=g==group
            alphas=tune(xx[~test],y[~test],g[~test])
            model=tuned_train(xx[~test],y[~test],alphas)
            oof[test]=predict(xx[test],model)
            results['folds'].append({'test_match':group,'model':name,'test_windows':int(test.sum()),
                                     'alphas':alphas.tolist(), **error_stats(wrap(oof[test]-y[test]))})
        predictions[name]=oof
        alphas=tune(xx,y,g)
        model=tuned_train(xx,y,alphas)
        results['final_models'][name]={'alphas':alphas.tolist(),'feature_names':FEATURES[:count],
            'coefficient_rows_features_columns_angles':model[0].tolist(),
            'intercept_deg':model[1].tolist(),
            'in_sample':error_stats(wrap(predict(xx,model)-y))}
    for name,p in predictions.items():
        results['models'][name]=error_stats(wrap(p-y))
    results['by_match_and_aircraft']=[]
    for file in FILES:
        for uid in UIDS:
            sel=(g==file)&(d['uid']==uid)
            results['by_match_and_aircraft'].append({'match':file,'uid':uid,'windows':int(sel.sum()),
                'models':{name:error_stats(wrap(p[sel]-y[sel])) for name,p in predictions.items()}})
    return results,predictions


def geometry(d):
    vdir=direction(d['velocity']); ndir=direction(d['nose'])
    pdir=direction(d['displacement']); mdir=direction(d['midpoint_velocity'])
    # Point-to-interval comparisons are labelled separately from midpoint diagnostics.
    derived={'nose_minus_velocity_heading':wrap(ndir[:,0]-vdir[:,0]),
             'nose_minus_velocity_elevation':ndir[:,1]-vdir[:,1],
             'nose_velocity_separation':separation(d['nose'],d['velocity']),
             'velocity_minus_displacement_heading':wrap(vdir[:,0]-pdir[:,0]),
             'velocity_minus_displacement_elevation':vdir[:,1]-pdir[:,1],
             'midpoint_velocity_minus_displacement_heading':wrap(mdir[:,0]-pdir[:,0]),
             'midpoint_velocity_minus_displacement_elevation':mdir[:,1]-pdir[:,1],
             'midpoint_velocity_displacement_separation':separation(d['midpoint_velocity'],d['displacement']),
             'roll':wrap(d['current'][:,0]),
             'track_turn_rate':wrap(direction(d['next_velocity'])[:,0]-vdir[:,0])/DT}
    stats={k:scalar_stats(v) for k,v in derived.items() if k not in ('roll','track_turn_rate')}
    stats['roll_track_turn_rate_pearson']=float(np.corrcoef(derived['roll'],derived['track_turn_rate'])[0,1])
    stats['by_match']={file:{k:scalar_stats(v[d['match']==file]) for k,v in derived.items()
                           if k not in ('roll','track_turn_rate')} for file in FILES}
    return stats,derived,vdir


def plots(d,pred,geo,vdir):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':.2,'figure.dpi':140})
    full=pred['history_and_velocity']
    for file in FILES:
        fig,axs=plt.subplots(4,3,figsize=(16,11),sharex=True,layout='constrained')
        for i,uid in enumerate(UIDS):
            sel=(d['match']==file)&(d['uid']==uid)
            t=d['time'][sel]+DT
            for j,angle in enumerate(ANGLES):
                ax=axs[i,j]
                ax.scatter(t,d['next'][sel,j],s=10,c='#222222',alpha=.6,label='Observed',zorder=3)
                ax.plot(t,d['current'][sel,j]+full[sel,j],c='#e45b22',lw=1.3,label='Predicted: held-out match',zorder=4)
                ax.set_title(f'{uid} | {angle}',loc='left',fontsize=11)
                ax.set_ylabel('Unwrapped angle (deg)')
                if i==3: ax.set_xlabel('Target time t + 0.2 s')
        axs[0,0].legend(fontsize=8)
        fig.suptitle(file.removesuffix('_actions.csv')+'\n0.2-second attitude prediction from past attitude and velocity',fontsize=15)
        fig.supxlabel('Each match predicted using only the other two matches; ridge tuning also excludes this match. No action inputs.',fontsize=9)
        fig.savefig(HERE/(file.removesuffix('_actions.csv')+'_attitude_overlay.png'))
        plt.close(fig)
    file=FILES[2]
    sel=(d['match']==file)&(d['uid']=='A0100')
    t=d['time'][sel]+DT
    fig,axs=plt.subplots(3,2,figsize=(14,10),layout='constrained')
    for j,angle in enumerate(ANGLES):
        ax=axs[j,0]; zoom=t<=20
        ax.scatter(t[zoom],d['next'][sel,j][zoom],s=18,c='#222222',alpha=.7,label='Observed',zorder=3)
        ax.plot(t[zoom],(d['current'][sel,j]+full[sel,j])[zoom],c='#e45b22',lw=1.6,label='Predicted: held-out match',zorder=4)
        ax.set_title(f'{angle}: first 20 s',loc='left');ax.set_ylabel('Unwrapped angle (deg)')
        ax.set_xlabel('Target time (s)')
        ax=axs[j,1]
        ax.axhline(0,color='#777777',lw=.7)
        for name,color,title in [('constant_rate','#4779ac','Constant angular rate'),
                                  ('history_and_velocity','#e45b22','History + velocity')]:
            ax.plot(t,wrap(pred[name][sel,j]-d['y'][sel,j]),lw=1,color=color,label=title)
        ax.set_title(f'{angle}: full-interval prediction error',loc='left')
        ax.set_ylabel('Predicted - observed (deg)');ax.set_xlabel('Target time (s)')
    axs[0,0].legend(fontsize=8);axs[0,1].legend(fontsize=8)
    fig.suptitle('A0100 | ours / ic100606 / win\nAttitude prediction evaluated on an unseen match',fontsize=16)
    fig.supxlabel('Predictions use data through t only; target is t + 0.2 s. These are one-step predictions, not a free-running rollout.',fontsize=9)
    fig.savefig(HERE/'attitude_detail_A0100_ic100606.png');plt.close(fig)

    fig,axs=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    colors=['#4779ac','#e45b22','#27977a']
    labels=['MHI / 100601 / win','Ours / 100601 / loss','Ours / 100606 / win']
    for file,color,label in zip(FILES,colors,labels):
        m=d['match']==file
        axs[0,0].scatter(d['time'][m],geo['nose_minus_velocity_heading'][m],s=5,alpha=.3,color=color,label=label)
        axs[0,1].scatter(vdir[m,1],d['current'][m,1],s=7,alpha=.3,color=color)
        axs[0,2].hist(geo['nose_velocity_separation'][m],bins=np.arange(0,np.ceil(geo['nose_velocity_separation'].max())+1,.5),histtype='step',lw=1.5,color=color,label=label)
        axs[1,0].scatter(geo['roll'][m],geo['track_turn_rate'][m],s=7,alpha=.3,color=color)
        axs[1,1].scatter(d['time'][m],geo['midpoint_velocity_minus_displacement_heading'][m],s=5,alpha=.3,color=color)
        axs[1,2].scatter(d['time'][m],geo['midpoint_velocity_minus_displacement_elevation'][m],s=5,alpha=.3,color=color)
    axs[0,0].set(xlabel='Time (s), all aircraft',ylabel='Yaw - velocity heading (deg)',title='Horizontal nose / motion mismatch')
    axs[0,0].legend(fontsize=8)
    lim=[min(vdir[:,1].min(),d['current'][:,1].min())-2,max(vdir[:,1].max(),d['current'][:,1].max())+2]
    axs[0,1].plot(lim,lim,'--',color='#777777',lw=1)
    axs[0,1].set(xlabel='Velocity flight-path angle (deg)',ylabel='Pitch (deg)',title='Pitch vs direction of travel',xlim=lim,ylim=lim)
    axs[0,2].set(xlabel='3D angular separation (deg)',ylabel='Window count',title='Nose vs velocity direction')
    axs[1,0].set(xlabel='Roll (deg)',ylabel='Velocity-heading rate (deg/s)',title='Bank and observed turn rate')
    axs[1,1].set(xlabel='Time (s), all aircraft',ylabel='Heading difference (deg)',title='Midpoint velocity vs displacement')
    axs[1,2].set(xlabel='Time (s), all aircraft',ylabel='Elevation difference (deg)',title='Midpoint velocity vs displacement')
    fig.suptitle('Attitude, velocity and position-change directions | all three matches',fontsize=16)
    fig.supxlabel('Assumed yaw: clockwise from North; pitch: nose elevation. WGS84 positions. Midpoint uses v(t + 0.2 s) for diagnostics only.',fontsize=9)
    fig.savefig(HERE/'direction_relationships.png');plt.close(fig)


def main():
    d,sources=dataset()
    result,pred=prediction_analysis(d)
    geom,derived,vdir=geometry(d)
    result.update({'windows':len(d['y']),'sources':sources,'direction_comparison':geom,
      'method':{
        'equation':'angle_next = angle_current + intercept + feature_vector @ coefficients; prediction errors wrapped to [-180,180).',
        'model':'Three ridge linear regressions, one per Euler-angle increment. Coefficients shared by all aircraft; no aircraft or match identifiers used as features.',
        'history':'Angles at n,n-1,n-2 encoded as two wrapped increments and sin/cos of current angles.',
        'velocity':'vn,ve,vu and TAS at n plus two backward differences, using n,n-1,n-2 only.',
        'split':'Outer leave-one-match-out; alpha per output chosen by inner leave-one-training-match-out. Scaling/intercept fitted using training data only.',
        'final_model':'Full-data models are saved for reuse. All prediction plots and headline errors use outer held-out predictions, not full-data models.',
        'baseline':'Persistence and constant-rate baseline; history-only ridge has the same nested validation to assess added velocity information.',
        'limitations':'Only three matches, two share ic100601. Target action at n is not included. One-step conditional prediction, not autonomous rollout or physical dynamics identification.',
        'angle_convention':'Assume yaw clockwise from North, positive pitch upward. README does not fully specify Euler conventions. Pitch observed stays away from +/-90 degrees.',
        'direction':'Forward axis = (cos(pitch)cos(yaw), cos(pitch)sin(yaw), sin(pitch)) in NEU. Roll does not change this forward axis.',
        'position':'WGS84 geodetic latitude and ellipsoidal altitude assumed, converted to ECEF; displacement and midpoint velocity projected to NEU at n.',
        'midpoint':'(v_ECEF[n]+v_ECEF[n+1])/2 only for direction diagnostics. Never used in prediction features.',
        'nose_warning':'Nose/velocity mismatch is not labelled aerodynamic angle of attack or sideslip: wind and air-relative vector are not established.',
        'filter':'Same four consecutive alive samples per aircraft as position fit. No residual-based exclusions.'}})
    (HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    with (HERE/'heldout_predictions.csv').open('w') as f:
        writer=csv.writer(f)
        writer.writerow(['match','uid','time_s','target_time_s']+[f'{a}_observed_unwrapped_deg' for a in ANGLES]
                        +[f'{a}_predicted_unwrapped_deg' for a in ANGLES]+[f'{a}_error_deg' for a in ANGLES])
        p=d['current']+pred['history_and_velocity'];e=wrap(p-d['next'])
        for i in range(len(e)):
            writer.writerow([d['match'][i],d['uid'][i],d['time'][i],d['time'][i]+DT,*d['next'][i],*p[i],*e[i]])
    plots(d,pred,derived,vdir)
    lines=['姿勢角予測と機首・移動方向の比較（2026-10-13 JST）',
           f'対象：3試合・各4機。連続生存区間 {len(d["y"])} 件。',
           '姿勢角ごとに1つの回帰式。機体別の係数は使用しない。',
           '全ての予測図は、対象試合を除く2試合だけで学習した予測。ハイパーパラメータ選択も対象試合を除外。',
           '全データで学習した再利用用係数は results.json の final_models。図には使用していない。',
           '目的変数は次の0.2秒間の角度増分。入力は現在と過去2時刻の姿勢角および速度（北・東・上、TAS）。',
           '角度誤差は±180度に折り返す。表示ではヨーの360度境界を連続化。',
           '\n試合外検証 RMSE [Roll, Pitch, Yaw]（度）:']
    for name,m in result['models'].items(): lines.append(f'{name}: {m["rmse_deg"]}')
    lines+=['\n今回の結果:',
            '姿勢角＋速度モデルは、直前の角速度を維持する外挿より3成分ともRMSEが小さい。',
            'ただし姿勢角だけの回帰に速度を追加しても明確な改善はない。ロールは悪化、ピッチは僅かに改善、ヨーはほぼ同じ。',
            'この結果は今回の特徴量・ridge回帰についての結果であり、速度一般に予測情報がないとは結論しない。',
            f'機首方向と速度方向の3D角度差：中央値 {geom["nose_velocity_separation"]["median"]:.4f} 度、95パーセンタイル {geom["nose_velocity_separation"]["p95_abs"]:.4f} 度。',
            f'区間両端速度平均と位置差分方向の3D角度差：中央値 {geom["midpoint_velocity_displacement_separation"]["median"]:.4f} 度。',
            f'ロールと次の0.2秒間の速度方位変化率の標本相関：{geom["roll_track_turn_rate_pearson"]:.4f}。時系列標本は独立ではなく、因果関係を同定した結果ではない。',
            '\n図:',
            'attitude_detail_A0100_ic100606.png：A0100の拡大図。黒点は実測、橙線は対象試合を除外したモデルの予測。右列は残差。',
            '*_attitude_overlay.png：3試合それぞれの全4機の実測・予測重ね合わせ。',
            'direction_relationships.png：姿勢・速度・位置差分方向の比較。',
            'heldout_predictions.csv：機体・時刻ごとの実測、試合外予測、誤差。',
            '\n方向の関係（度、ただし相関係数を除く）:',json.dumps({k:v for k,v in geom.items() if k!='by_match'},ensure_ascii=False,indent=2),
            '\n解釈・前提:',
            '機首方向と速度方向は別の量。ロールは機首の前方軸そのものを変えないが旋回との関連を散布図で確認する。',
            '速度方向と位置差分方向は、点の速度と区間平均の違いがある。両端速度平均との比較は診断用で、未来予測には使わない。',
            '姿勢角の規約、風、速度の厳密な定義はREADMEだけでは確定できない。機首と移動方向の差を迎角・横滑り角と断定しない。',
            '位置変換にはWGS84を仮定。3試合中2試合は同じ初期配置であり、独立な初期配置への一般化は未確認。',
            '角度予測は実測された過去状態を毎回入力する1ステップ予測。自律的な長期ロールアウトの精度ではない。',
            '禁止された元メモおよび外部コードは未読。']
    (HERE/'summary.txt').write_text('\n'.join(lines)+'\n')
    # Check the OLS implementation against numpy's direct least squares.
    x=d['x'][:,:12]; y=d['y']
    direct=np.linalg.lstsq(np.column_stack((np.ones(len(x)),x)),y,rcond=None)[0]
    assert np.allclose(predict(x,train(x,y,0)),np.column_stack((np.ones(len(x)),x))@direct,atol=1e-8)
    assert len(d['y'])==3601
    print(json.dumps({'windows':len(d['y']),'models':result['models'],
                      'direction':{k:v for k,v in geom.items() if k!='by_match'}},indent=2))


if __name__=='__main__': main()
