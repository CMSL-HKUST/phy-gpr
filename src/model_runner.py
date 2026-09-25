"""Reviewer-derived 11-model AlSi10Mg GP runner."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm


def _gp(x, y, tune, alpha, seed, restarts=4):
    ker = ConstantKernel(1., (0.1, 10.))*RBF(np.ones(x.shape[1]), (0.15, 8.)) + WhiteKernel(1e-3, (1e-4, .5))
    xs, ys = StandardScaler().fit(x), StandardScaler().fit(y[:, None])
    m = GaussianProcessRegressor(kernel=ker, alpha=alpha, normalize_y=False,
        optimizer='fmin_l_bfgs_b' if tune else None, n_restarts_optimizer=restarts if tune else 0, random_state=seed)
    m.fit(xs.transform(x), ys.transform(y[:, None]).ravel())
    return xs, ys, m


def _pred(m, x):
    xs, ys, gp = m; z, s = gp.predict(xs.transform(x), return_std=True)
    return ys.inverse_transform(z[:, None]).ravel(), s*ys.scale_[0]


def _step1(frame, tune, seed, restarts):
    x=frame[['P','v']].to_numpy(float); out={}
    for key, transform in [('d_mean',lambda z:z),('phi_mean',lambda z:np.log(np.maximum(z,1e-3)))]:
        target=key.split('_')[0]; out[key]=_gp(x,transform(frame[key].to_numpy(float)),tune,1e-8,seed,restarts=restarts)
    for key in ('d_std','phi_std'): out[key]=_gp(x,np.log(np.maximum(frame[key].to_numpy(float),1e-8)),tune,1e-8,seed,restarts=restarts)
    return out


def _features(s1,x):
    dm,_=_pred(s1['d_mean'],x); dlog,_=_pred(s1['d_std'],x); pm,_=_pred(s1['phi_mean'],x); plog,_=_pred(s1['phi_std'],x)
    d=np.maximum(dm,1e-6); ds=np.exp(dlog); h=d**-.5; hv=.25*d**-3*ds**2
    phi=np.exp(pm)/100.; ps=np.exp(plog)/100.; p=(1-phi)**2; pv=4*(1-phi)**2*ps**2
    return h,hv,p,pv


def _scale(ftr,fte):
    vals=[]
    for i in (0,2):
        c=float(np.mean(ftr[i])); s=max(float(np.std(ftr[i])),1e-8)
        vals += [(ftr[i]-c)/s,ftr[i+1]/s**2,(fte[i]-c)/s,fte[i+1]/s**2]
    return (vals[0],vals[1],vals[4],vals[5]),(vals[2],vals[3],vals[6],vals[7])


def _erbf(a,va,b,vb,l):
    v=np.maximum(va[:,None]+vb[None,:],1e-12); den=l*l+v
    return np.sqrt(l*l/den)*np.exp(-(a[:,None]-b[None,:])**2/(2*den))


def _kernel(xa,xb,fa,fb,q,active,diag):
    sp,sg,lp,lh,lq,sn=np.exp(q); h, hv, p, pv=fa; H, HV, P, PV=fb
    d=((xa[:,None,:]-xb[None,:,:])**2).sum(2)
    phys=np.ones_like(d)
    if 'h' in active: phys*= _erbf(h,hv,H,HV,lh)
    if 'p' in active: phys*= _erbf(p,pv,P,PV,lq)
    k=sp**2*np.exp(-d/(2*lp**2))+sg**2*phys
    if diag is not None: k+=np.diag(sn**2+diag)
    return k


def _physics(x,y,f,alpha,tune,active,seed,restarts):
    q0=np.log([1,1,1,1,1,.1]); bounds=[(-4,4)]*5+[(-7,1)]
    def nll(q):
        try:
            c=cho_factor(_kernel(x,x,f,f,q,active,alpha),lower=True,check_finite=False); a=cho_solve(c,y,check_finite=False)
            return float(.5*y@a+np.log(np.diag(c[0])).sum()+.5*len(y)*np.log(2*np.pi))
        except np.linalg.LinAlgError:return 1e12
    q=q0
    if tune:
        rng=np.random.default_rng(seed); starts=[q0]+[np.array([rng.uniform(a,b) for a,b in bounds]) for _ in range(max(0,restarts-1))]
        q=min((minimize(nll,z,method='L-BFGS-B',bounds=bounds) for z in starts),key=lambda z:z.fun).x
    c=cho_factor(_kernel(x,x,f,f,q,active,alpha),lower=True,check_finite=False)
    return {'x':x,'y':y,'f':f,'q':q,'active':active,'c':c}


def _pp(m,x,f):
    k=_kernel(x,m['x'],f,m['f'],m['q'],m['active'],None); return k@cho_solve(m['c'],m['y'],check_finite=False)


def _pp_with_std(m, x, f):
    k = _kernel(x, m['x'], f, m['f'], m['q'], m['active'], None)
    mean = k @ cho_solve(m['c'], m['y'], check_finite=False)
    v = cho_solve(m['c'], k.T, check_finite=False)
    kss = _kernel(x, x, f, f, m['q'], m['active'], None)
    std = np.sqrt(np.maximum(np.diag(kss) - (k * v.T).sum(axis=1), 1e-12))
    return mean, std


def _gaussian_crps(y, mean, std):
    std = np.maximum(np.asarray(std, float), 1e-12)
    z = (np.asarray(y, float) - np.asarray(mean, float)) / std
    return std * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))


def _det_features(s1, x):
    d, _ = _pred(s1['d_mean'], x)
    phi_log, _ = _pred(s1['phi_mean'], x)
    return np.column_stack([np.maximum(d, 1e-6), np.maximum(np.exp(phi_log), 1e-6)])


def _det_kernel(xa, xb, za, zb, q, diag=None):
    sp, sg, lp, ld, lphi, sn = np.exp(q)
    dx = ((xa[:, None, :] - xb[None, :, :]) ** 2).sum(2)
    dd = ((za[:, None, 0] - zb[None, :, 0]) ** 2) / (2 * ld**2)
    dp = ((za[:, None, 1] - zb[None, :, 1]) ** 2) / (2 * lphi**2)
    k = sp**2 * np.exp(-dx / (2 * lp**2)) + sg**2 * np.exp(-(dd + dp))
    if diag is not None:
        k += np.diag(sn**2 + diag)
    return k


def _det_physics(x, y, z, alpha, tune, seed, restarts):
    q0 = np.log([1, 1, 1, 1, 1, .1]); bounds = [(-4, 4)]*5 + [(-7, 1)]
    def nll(q):
        try:
            c = cho_factor(_det_kernel(x, x, z, z, q, alpha), lower=True, check_finite=False)
            a = cho_solve(c, y, check_finite=False)
            return float(.5*y@a + np.log(np.diag(c[0])).sum() + .5*len(y)*np.log(2*np.pi))
        except np.linalg.LinAlgError:
            return 1e12
    q = q0
    if tune:
        rng = np.random.default_rng(seed)
        starts = [q0] + [np.array([rng.uniform(a, b) for a, b in bounds]) for _ in range(max(0, restarts-1))]
        q = min((minimize(nll, z0, method='L-BFGS-B', bounds=bounds) for z0 in starts), key=lambda r:r.fun).x
    c = cho_factor(_det_kernel(x, x, z, z, q, alpha), lower=True, check_finite=False)
    return {'x':x, 'y':y, 'z':z, 'q':q, 'c':c}


def _det_pp(m, x, z):
    k = _det_kernel(x, m['x'], z, m['z'], m['q'], None)
    return k @ cho_solve(m['c'], m['y'], check_finite=False)


def _det_pp_with_std(m, x, z):
    k = _det_kernel(x, m['x'], z, m['z'], m['q'], None)
    mean = k @ cho_solve(m['c'], m['y'], check_finite=False)
    v = cho_solve(m['c'], k.T, check_finite=False)
    kss = _det_kernel(x, x, z, z, m['q'], None)
    return mean, np.sqrt(np.maximum(np.diag(kss) - (k * v.T).sum(axis=1), 1e-12))


def run(df, ng, seed, repeats, test_groups, diag_alpha, out, restarts=5):
    rng=np.random.default_rng(seed); rows=[]; step1_rows=[]
    for rep in range(1,repeats+1):
        ids=rng.choice(df.group_id.to_numpy(),test_groups,replace=False); tr=df[~df.group_id.isin(ids)].reset_index(drop=True); te=df[df.group_id.isin(ids)].reset_index(drop=True)
        xtr=tr[['P','v']].to_numpy(float); xte=te[['P','v']].to_numpy(float); ytr=tr.yield_mean.to_numpy(float); yte=te.yield_mean.to_numpy(float)
        s1=_step1(tr,True,seed,restarts)
        for split, frame, x in [('train',tr,xtr),('test',te,xte)]:
            for target, key, transform, inverse in [
                ('d','d_mean',lambda z:z,lambda z:z),
                ('phi','phi_mean',lambda z:np.log(np.maximum(z,1e-3)),lambda z:np.exp(z)),
            ]:
                mean_pred,_=_pred(s1[key],x); mean_pred=inverse(mean_pred)
                std_pred,_=_pred(s1[f'{target}_std'],x); std_pred=np.exp(std_pred)
                for stat, truth, pred in [('mean',frame[f'{target}_mean'].to_numpy(float),mean_pred),('std',frame[f'{target}_std'].to_numpy(float),std_pred)]:
                    rows_metrics={'repeat':rep,'split':split,'target':target,'stat':stat,'mae':mean_absolute_error(truth,pred),'rmse':mean_squared_error(truth,pred)**.5}
                    step1_rows.append(rows_metrics)
        ft,fe=_scale(_features(s1,xtr),_features(s1,xte)); s1_all=_step1(df,True,seed,restarts); fb, fbe=_scale(_features(s1_all,xtr),_features(s1_all,xte)); xs,ys=StandardScaler().fit(xtr),StandardScaler().fit(ytr[:,None]); zx,zq=xs.transform(xtr),xs.transform(xte); zy=ys.transform(ytr[:,None]).ravel(); alpha=(tr.yield_std.to_numpy()/np.sqrt(ng)/ys.scale_[0])**2 if diag_alpha else np.zeros(len(tr))
        models={'untuned_physics_informed':(ft,fe,False,('h','p')),'untuned_physics_h_only':(ft,fe,False,('h',)),'untuned_physics_p_only':(ft,fe,False,('p',)),'untuned_physics_better_upstream':(fb,fbe,False,('h','p')),'tuned_physics_informed':(ft,fe,True,('h','p'))}
        vals={}
        for name,(a,b,tune,act) in models.items():
            m=_physics(zx,zy,a,alpha,tune,act,seed,restarts)
            ma,sa=_pp_with_std(m,zx,a); mb,sb=_pp_with_std(m,zq,b)
            vals[name]=(ys.inverse_transform(ma[:,None]).ravel(),ys.inverse_transform(mb[:,None]).ravel(),sa*ys.scale_[0],sb*ys.scale_[0])
        ztr_det = _det_features(s1, xtr); zte_det = _det_features(s1, xte)
        for tune in (False, True):
            tag = 'tuned' if tune else 'untuned'
            m = _det_physics(zx, zy, ztr_det, alpha, tune, seed, restarts)
            da, dsa = _det_pp_with_std(m, zx, ztr_det); db, dsb = _det_pp_with_std(m, zq, zte_det)
            vals[f'{tag}_physics_mean_only'] = (
                ys.inverse_transform(da[:, None]).ravel(), ys.inverse_transform(db[:, None]).ravel(),
                dsa*ys.scale_[0], dsb*ys.scale_[0],
            )
        for tune in (False,True):
            tag='tuned' if tune else 'untuned'
            for name,a,b in ((tag+'_process_only',xtr,xte),(tag+'_d_phi_only',tr[['d_mean','phi_mean']].to_numpy(float),te[['d_mean','phi_mean']].to_numpy(float))):
                m=_gp(a,ytr,tune,alpha,seed,restarts=restarts); pa,sa=_pred(m,a); pb,sb=_pred(m,b); vals[name]=(pa,pb,sa,sb)
        for name,(a,b,sa,sb) in vals.items():
            rows.extend([
                {'repeat':rep,'split':'train','model':name,'mae':mean_absolute_error(ytr,a),'rmse':mean_squared_error(ytr,a)**.5,'crps':float(np.mean(_gaussian_crps(ytr,a,sa)))},
                {'repeat':rep,'split':'test','model':name,'mae':mean_absolute_error(yte,b),'rmse':mean_squared_error(yte,b)**.5,'crps':float(np.mean(_gaussian_crps(yte,b,sb)))},
            ])
    out=Path(out);out.mkdir(parents=True,exist_ok=True); pd.DataFrame(step1_rows).to_csv(out/'step1_metrics.csv',index=False); u=pd.DataFrame(step1_rows).groupby(['split','target','stat'])[['mae','rmse']].agg(['mean','std']).reset_index();u.columns=['_'.join(c).strip('_') for c in u.columns];u.to_csv(out/'step1_summary.csv',index=False); pd.DataFrame(rows).to_csv(out/'metrics_by_repeat.csv',index=False); s=pd.DataFrame(rows).groupby(['split','model'])[['mae','rmse','crps']].agg(['mean','std']).reset_index();s.columns=['_'.join(c).strip('_') for c in s.columns];s=s.sort_values(['split','mae_mean']);s.to_csv(out/'summary_sorted_by_mae.csv',index=False);print('Step1\n',u.to_string(index=False));print('Step2\n',s.to_string(index=False)); return {"step1_summary": u, "summary": s, "output_dir": out}
