"""Plot observed step displacements and fitted predictions, in local NEU axes."""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fit import HERE, RAW, FILES, DT, AB3, coordinates

COEF = np.array(json.loads((HERE/'results.json').read_text())['wgs84']['models']['sum1']['coefficients'])
plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                     'axes.grid':True, 'grid.alpha':.2, 'figure.dpi':140})
AXES = ['North', 'East', 'Up']


def series(rows, uid):
    rr = sorted((r for r in rows if r['uid']==uid), key=lambda r:int(r['step']))
    pos, vel = coordinates(rr,'wgs84')
    times, obs, pred, ab = [],[],[],[]
    for n in range(2,len(rr)-1):
        window=rr[n-2:n+2]
        if not all(r['alive']=='1' for r in window):
            continue
        if not np.all(np.diff([int(r['step']) for r in window])==1):
            continue
        if not np.allclose(np.diff([float(r['time_s']) for r in window]),DT,rtol=0,atol=1e-9):
            continue
        lat,lon=np.deg2rad([float(rr[n]['lat_deg']),float(rr[n]['lon_deg'])])
        rotation=np.array([[-np.sin(lat)*np.cos(lon),-np.sin(lat)*np.sin(lon),np.cos(lat)],
                           [-np.sin(lon),np.cos(lon),0],
                           [np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)]])
        x=np.stack([vel[n],vel[n-1],vel[n-2]],axis=-1)
        times.append(float(rr[n]['time_s']))
        obs.append(rotation@(pos[n+1]-pos[n]))
        pred.append(rotation@(DT*x@COEF))
        ab.append(rotation@(DT*x@AB3))
    return tuple(np.array(v) for v in (times,obs,pred,ab))


def main():
    all_errors=[]
    for file in FILES:
        with (RAW/file).open() as f:
            rows=list(csv.DictReader(f))
        fig,axs=plt.subplots(4,3,figsize=(16,11),sharex=True,layout='constrained')
        for i,uid in enumerate(['A0100','A0200','B0100','B0200']):
            t,y,p,ab=series(rows,uid)
            all_errors.append(p-y)
            for j,axis in enumerate(AXES):
                ax=axs[i,j]
                ax.scatter(t,y[:,j],s=10,c='#222222',alpha=.6,label='Observed',zorder=3)
                ax.plot(t,p[:,j],c='#e45b22',lw=1.5,label='Fitted (sum = 1)',zorder=4)
                ax.set_title(f'{uid} | {axis}',loc='left',fontsize=11)
                ax.set_ylabel('Displacement / 0.2 s (m)')
                if i==3: ax.set_xlabel('Time t (s)')
        axs[0,0].legend(loc='best',fontsize=8)
        fig.suptitle(file.removesuffix('_actions.csv')+'\nObserved step displacement vs fitted prediction',fontsize=16)
        fig.supxlabel('Shared coefficients fitted on all three matches (in-sample). WGS84 assumed; no attitude-angle fit.',fontsize=10)
        stem=file.removesuffix('_actions.csv')
        fig.savefig(HERE/f'{stem}_overlay.png')
        plt.close(fig)
        if '100606' in file:
            t,y,p,ab=series(rows,'A0100')
            fig,axs=plt.subplots(3,2,figsize=(14,10),layout='constrained')
            for j,axis in enumerate(AXES):
                ax=axs[j,0]
                zoom=t<=20
                ax.scatter(t[zoom],y[zoom,j],s=18,c='#222222',alpha=.7,label='Observed',zorder=3)
                ax.plot(t[zoom],p[zoom,j],c='#e45b22',lw=1.6,label='Fitted (sum = 1)',zorder=4)
                ax.set_title(f'{axis}: observed and fitted (first 20 s)',loc='left')
                ax.set_ylabel('Displacement / 0.2 s (m)')
                ax.set_xlabel('Time t (s)')
                ax=axs[j,1]
                ax.axhline(0,color='#777777',lw=.8)
                ax.plot(t,ab[:,j]-y[:,j],c='#4779ac',alpha=.8,lw=1,label='AB3 residual')
                ax.plot(t,p[:,j]-y[:,j],c='#e45b22',lw=1,label='Fitted residual')
                ax.set_title(f'{axis}: residual over full alive interval',loc='left')
                ax.set_ylabel('Prediction - observation (m)')
                ax.set_xlabel('Time t (s)')
            axs[0,0].legend(fontsize=9)
            axs[0,1].legend(fontsize=9)
            fig.suptitle('A0100 | ours / ic100606 / win\nPosition-displacement fit: overlay and residuals',fontsize=16)
            fig.supxlabel('a = (1.683695, -0.916378, 0.232683), dt = 0.2 s. In-sample fit; WGS84 assumed.',fontsize=10)
            fig.savefig(HERE/'detail_A0100_ic100606.png')
            plt.close(fig)
    errors=np.concatenate(all_errors)
    expected=json.loads((HERE/'results.json').read_text())['wgs84']['models']['sum1']['rmse_m']
    actual=float(np.sqrt(np.mean(np.sum(errors**2,axis=1))))
    assert len(errors)==3601
    assert abs(actual-expected)<1e-10, (actual,expected)
    print(f'Created 4 figures. Verified {len(errors)} windows; RMSE={actual:.9f} m.')


if __name__=='__main__':
    main()
