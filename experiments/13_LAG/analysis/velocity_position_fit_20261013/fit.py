"""Fit three velocity lags; read only the three explicitly named action CSVs.

Run with Python + numpy. Outputs results.json and summary.txt beside this script.
Coordinates assume WGS84 geodetic latitude and ellipsoidal height. Sphere is a
sensitivity check, not a claim about the simulator. No simulator code is read.
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE.parents[1] / 'data/raw/20261005_LAG_2v2_ACMI/actions'
FILES = ['mhi_v1.3_ic100601_win_actions.csv',
         'ours_seed100100_ic100601_loss_actions.csv',
         'ours_seed100100_ic100606_win_actions.csv']
DT = 0.2
AB3 = np.array([23, -16, 5]) / 12


def coordinates(rows, model):
    lat, lon, h = np.array([[float(r[k]) for k in
                           ['lat_deg', 'lon_deg', 'alt_m']] for r in rows]).T
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    a, e2 = (6378137., 6.6943799901413165e-3) if model == 'wgs84' else (6371000., 0.)
    radius = a / np.sqrt(1 - e2 * sl**2)
    pos = np.column_stack(((radius+h)*cl*co, (radius+h)*cl*so,
                           (radius*(1-e2)+h)*sl))
    neu = np.array([[float(r[k]) for k in ['vn_mps','ve_mps','vu_mps']] for r in rows])
    north = np.column_stack((-sl*co, -sl*so, cl))
    east = np.column_stack((-so, co, np.zeros(len(rows))))
    up = np.column_stack((cl*co, cl*so, sl))
    vel = neu[:,0,None]*north + neu[:,1,None]*east + neu[:,2,None]*up
    if model == 'vertical':
        return h[:,None], neu[:,2,None]
    return pos, vel


def dataset(model):
    xx, yy, vv, labels, audits = [], [], [], [], []
    for name in FILES:
        with (RAW/name).open() as f:
            rows = list(csv.DictReader(f))
        assert len({(r['uid'], r['step']) for r in rows}) == len(rows)
        counts = {}
        for uid in sorted({r['uid'] for r in rows}):
            rr = sorted((r for r in rows if r['uid']==uid), key=lambda r: int(r['step']))
            p, v = coordinates(rr, model)
            valid = 0
            for n in range(2, len(rr)-1):
                window = rr[n-2:n+2]
                if not all(r['alive']=='1' for r in window):
                    continue
                if not np.allclose(np.diff([float(r['time_s']) for r in window]), DT, atol=1e-9, rtol=0):
                    continue
                if not np.all(np.diff([int(r['step']) for r in window]) == 1):
                    continue
                x = np.stack([v[n], v[n-1], v[n-2]], axis=-1)
                y = (p[n+1]-p[n])/DT
                assert np.isfinite(x).all() and np.isfinite(y).all()
                xx.append(x); yy.append(y); vv.append(v[n+1]); labels.append(name)
                valid += 1
            counts[uid] = valid
        audits.append({'file':name, 'rows':len(rows), 'windows_by_uid':counts,
                       'sha256':hashlib.sha256((RAW/name).read_bytes()).hexdigest()})
    return np.array(xx), np.array(yy), np.array(vv), np.array(labels), audits


def fit(x, y, constrained):
    x, y = x.reshape(-1,3), y.reshape(-1)
    if constrained:
        b = np.linalg.lstsq(x[:,1:]-x[:,0,None], y-x[:,0], rcond=None)[0]
        return np.r_[1-b.sum(), b]
    return np.linalg.lstsq(x,y,rcond=None)[0]


def metrics(error):
    e = np.linalg.norm(error*DT, axis=1)
    return {'rmse_m':float(np.sqrt(np.mean(e**2))),
            'median_m':float(np.median(e)), 'p95_m':float(np.quantile(e,.95)),
            'max_m':float(e.max())}


def analyze(model):
    x,y,vnext,label,audit = dataset(model)
    result = {'windows':len(y), 'audit':audit,
              'condition_number':float(np.linalg.cond(x.reshape(-1,3))),
              'models':{}, 'leave_one_match_out':[], 'per_match':[]}
    coefficients = {'Euler':np.array([1.,0,0]), 'AB3':AB3,
                    'free':fit(x,y,False), 'sum1':fit(x,y,True)}
    for name,c in coefficients.items():
        result['models'][name] = {'coefficients':c.tolist(), 'sum':float(c.sum()),
                                  **metrics(x@c-y)}
    result['trapezoid_diagnostic_future_velocity'] = metrics((x[:,:,0]+vnext)/2-y)
    heldout_errors = {name:[] for name in coefficients}
    for file in FILES:
        test = label == file
        for name in coefficients:
            c = fit(x[~test],y[~test],name=='sum1') if name in ('free','sum1') else coefficients[name]
            err = x[test]@c-y[test]
            heldout_errors[name].append(err)
            result['leave_one_match_out'].append({'test_file':file,'model':name,
                'coefficients':c.tolist(), **metrics(err)})
        for name in ('free','sum1'):
            c = fit(x[test],y[test],name=='sum1')
            result['per_match'].append({'file':file,'model':name,'coefficients':c.tolist(),
                                        **metrics(x[test]@c-y[test])})
    result['pooled_heldout'] = {k:metrics(np.concatenate(v)) for k,v in heldout_errors.items()}
    # Verify constraint and a known synthetic solution using the same design.
    assert abs(coefficients['sum1'].sum()-1)<1e-12
    for constrained in (False,True):
        assert np.allclose(fit(x,x@AB3,constrained),AB3,atol=1e-9)
    return result


def main():
    results = {model:analyze(model) for model in ['wgs84','vertical','sphere']}
    results['method'] = {
        'equation':'(r[n+1]-r[n])/0.2 = a0*v[n]+a1*v[n-1]+a2*v[n-2]',
        'objective':'Unweighted least squares over all valid windows and components; no intercept.',
        'coordinates':'WGS84 geodetic+ellipsoidal height -> ECEF; local NEU velocities rotated to ECEF at each sample.',
        'assumptions':'README does not establish geodetic versus geocentric latitude, altitude datum, or ground-relative velocity semantics.',
        'sensitivity':'Vertical uses altitude difference and vu only. Sphere uses R=6371000 m.',
        'evaluation':'One-step displacement Euclidean RMSE, not cumulative trajectory error. Match-level holdout is not independent initial-condition validation: two matches share ic100601.',
        'exclusions':'Four consecutive alive rows from one aircraft, steps n-2 to n+1, all three intervals 0.2 seconds. No residual-based outlier deletion.',
        'future_diagnostic':'Trapezoid uses v[n+1], for alignment/coordinate diagnostics only, not causal prediction.'}
    (HERE/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    lines = ['Velocity-position fit, 2026-10-13 JST', json.dumps(results['method'],indent=2)]
    for model in ['wgs84','vertical','sphere']:
        r = results[model]
        lines += [f'\n{model}: {r["windows"]} windows, condition number {r["condition_number"]:.3f}']
        for name,m in r['models'].items():
            lines.append(f'{name}: a={m["coefficients"]}, train RMSE={m["rmse_m"]:.9f} m, heldout RMSE={r["pooled_heldout"][name]["rmse_m"]:.9f} m')
        lines.append('Per-match fits: '+json.dumps(r['per_match']))
    text = '\n'.join(lines)+'\n'
    (HERE/'summary.txt').write_text(text)
    print(text)


if __name__ == '__main__':
    main()
