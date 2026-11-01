"""Opt-in paired ordinal DIF pilot; production APIs only, immutable cached results."""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = Path(__file__).with_name('plan.json')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def clean(value):
    if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple,np.ndarray)): return [clean(v) for v in value]
    if isinstance(value, (np.bool_,bool)): return bool(value)
    if isinstance(value, (np.integer,int)): return int(value)
    if isinstance(value, (np.floating,float)): return float(value) if np.isfinite(value) else None
    return value


def save(path, value):
    path.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False))


def binomial_ci(events, trials, alpha=.05):
    if trials == 0: return [0.,1.]
    return [float(beta.ppf(alpha/2,events,trials-events+1)) if events else 0.,
            float(beta.ppf(1-alpha/2,events+1,trials-events)) if events<trials else 1.]


def worst_deviation(n, alpha):
    k=np.arange(n+1); rate=k/n
    lo=beta.ppf(alpha/2,k,n-k+1); lo[0]=0
    hi=beta.ppf(1-alpha/2,k+1,n-k); hi[-1]=1
    return float(max(np.max(rate-lo),np.max(hi-rate)))


def precision_plan(plan):
    alpha=plan['confidence_family_alpha']/len(plan['methods'])
    low,high=1,2
    while worst_deviation(high,alpha)>plan['target_max_interval_deviation']: high*=2
    while low<high:
        mid=(low+high)//2
        if worst_deviation(mid,alpha)<=plan['target_max_interval_deviation']: high=mid
        else: low=mid+1
    return dict(required_repetitions=low,method_alpha=alpha,
                worst_ci_deviation=worst_deviation(low,alpha),
                previous_n_deviation=worst_deviation(low-1,alpha),
                worst_bernoulli_mcse=math.sqrt(.25/low))


def classify(p, alpha):
    from core.dif_classical import adjust_pvalues
    p=np.asarray(p,float)
    if p.ndim!=1 or len(p)!=2 or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        return dict(valid=False,reason='Incomplete/invalid common two-item p family')
    bh=adjust_pvalues(p,'BH'); holm=adjust_pvalues(p,'Holm')
    return dict(valid=True,p=p.tolist(),bh=bh.tolist(),holm=holm.tolist(),
                raw_rejections=(p<alpha).astype(int).tolist(),
                holm_event=bool(np.any(holm<alpha)),bh_event=bool(np.any(bh<alpha)))


def observed(data, group, plan, method):
    from core.dif_classical import mantel_ordinal, logistic_dif
    t=time.monotonic(); raw=[]
    try:
        match=data.sum(axis=1).to_numpy(); gf=np.asarray(group)
        for name in plan['tested_family']:
            y=data[name].to_numpy()
            raw.append(mantel_ordinal(y,gf,match) if method=='Mantel' else logistic_dif(y,gf,match,ordinal=True))
        p=[r['p'] if method=='Mantel' else r['p_omnibus'] for r in raw]
        result=classify(p,plan['alpha'])
        result.update(raw=raw,convergence='not_applicable' if method=='Mantel' else 'production convergence gate')
    except Exception as exc:
        result=dict(valid=False,reason=f'{type(exc).__name__}: {exc}')
    result['seconds']=time.monotonic()-t
    return result


def grm_worker(data_path, output_path):
    from core import grm_dif as G
    plan=json.loads(PLAN_PATH.read_text()); frame=pd.read_csv(data_path); group=frame.pop('group')
    t=time.monotonic()
    try:
        out=G.test_dif(frame,group,0,1,plan['anchors'],categories=plan['categories'],points=61,alpha=plan['alpha'])
        table=out['table'].set_index('Madde').loc[plan['tested_family']]
        result=classify(table['p'],plan['alpha'])
        result.update(table=table.reset_index().to_dict('records'),fits={},n_fits=out['n_fits'])
        for item,fits in out['fits'].items():
            result['fits'][frame.columns[item]]={}
            for label,f in fits.items():
                a,b,mu,sd=f['design'].unpack(f['params'])
                diag={k:v for k,v in f.items() if k not in ('design','params','cov')}
                diag.update(a=a,b=b,focal_mean=mu,focal_sd=sd,information_available=f['cov'] is not None)
                result['fits'][frame.columns[item]][label]=diag
    except Exception as exc:
        result=dict(valid=False,reason=f'{type(exc).__name__}: {exc}')
    result['seconds']=time.monotonic()-t
    save(output_path,result)


def score_mechanism(points, impact):
    """No fit: exact discrete total-score convolution, GH latent integration."""
    from numpy.polynomial.hermite_e import hermegauss
    z,w=hermegauss(points);w=w/np.sqrt(2*np.pi)
    a=np.linspace(.9,1.8,10);b=np.linspace(-1,1,10)[:,None]+np.array([-1.,0.,1.])
    result={}
    for j in (8,9):
        groups=[]
        for mu in (0.,impact):
            surv=expit(a[None,:,None]*(z[:,None,None]+mu-b[None,:,:]))
            probs=np.diff(np.concatenate([np.zeros((points,10,1)),1-surv,np.ones((points,10,1))],axis=2),axis=2)
            mass=np.zeros(31);num=np.zeros(31)
            for q in range(points):
                other=np.ones(1)
                for i in range(10):
                    if i!=j: other=np.convolve(other,probs[q,i])
                for c in range(4):
                    v=w[q]*probs[q,j,c]*other
                    mass[c:c+len(other)]+=v;num[c:c+len(other)]+=c*v
            groups.append(dict(total_probability=mass,conditional_mean=num/mass))
        result[f'P{j+1:02d}']=dict(groups=groups,mean_difference=groups[1]['conditional_mean']-groups[0]['conditional_mean'])
    return clean(result)


def summarize(records, plan, precision, main_completed=False):
    from validation.coverage_precision import coverage_precision
    result=dict(executed=len(records),planned_pilot=plan['pilot_repetitions'],methods={},paired={},
                scientific_status='PILOT ONLY / NOT CERTIFIED' if not main_completed else 'SCOPED EXPERIMENT',
                required_repetitions=precision['required_repetitions'])
    for method in plan['methods']:
        rows=[r['methods'][method] for r in records]; valid=[r for r in rows if r['valid']]
        n=len(valid);failed=len(rows)-n;s=sum(r['holm_event'] for r in valid);bh=sum(r['bh_event'] for r in valid)
        robust=[binomial_ci(s,len(rows),precision['method_alpha'])[0],
                binomial_ci(s+failed,len(rows),precision['method_alpha'])[1]]
        decision='INCONCLUSIVE'
        if main_completed:
            if robust[1]<=plan['alpha']:decision='SCOPED CONTROL SUPPORTED'
            elif robust[0]>plan['alpha']:decision='INFLATION SUPPORTED'
        row=dict(attempted=len(rows),valid=n,failed=failed,valid_rate=n/len(rows) if rows else None,
                 holm_fwer=s/n if n else None,holm_events=s,
                 holm_ci=binomial_ci(s,n,precision['method_alpha']),failure_robust_ci=robust,
                 failure_range=[s/len(rows),(s+failed)/len(rows)] if rows else [0,1],
                 holm_mcse=math.sqrt((s/n)*(1-s/n)/n) if n else None,
                 bh_fdr=bh/n if n else None,bh_fwer=bh/n if n else None,
                 bh_ci95=binomial_ci(bh,n),bh_mcse=math.sqrt((bh/n)*(1-bh/n)/n) if n else None,
                 decision=decision,interpretation='Conditional on complete valid families; complete null FDR=FWER')
        if method=='GRM_LR':
            fits=[[f for hypotheses in r.get('fits',{}).values() for f in hypotheses.values()] for r in rows]
            row['convergence_confirmed']=sum(bool(fs) and all(f['converged'] for f in fs) for fs in fits)
            row['convergence_not_assessed']=sum(not fs for fs in fits)
        elif method=='ordinal_LR':
            row['convergence_confirmed']=n  # production returns p only after all three regressions converge
            row['convergence_not_confirmed']=failed
        else:
            row['convergence']='not_applicable: stratified statistic'
        if n:
            parts=[np.array(r['raw_rejections'])[:,None] for r in valid]
            row['raw_type1']=float(np.mean([r['raw_rejections'] for r in valid]))
            row['raw_precision']=coverage_precision(parts,n)
        result['methods'][method]=row
    for a,b in itertools.combinations(plan['methods'],2):
        pairs=[(r['methods'][a],r['methods'][b]) for r in records if r['methods'][a]['valid'] and r['methods'][b]['valid']]
        d=np.array([int(x['holm_event'])-int(y['holm_event']) for x,y in pairs]);n=len(d);missing=len(records)-n
        eps=math.sqrt(2*math.log(2/.05)/n) if n else 2
        result['paired'][a+' minus '+b]=dict(complete_pairs=n,missing_pairs=missing,
            a_only=int(np.sum(d==1)),b_only=int(np.sum(d==-1)),
            mean=float(d.mean()) if n else None,mcse=float(d.std(ddof=1)/math.sqrt(n)) if n>1 else None,
            marginal_ci95=[max(-1,float(d.mean())-eps),min(1,float(d.mean())+eps)] if n else [-1,1],
            missing_pair_range=[float((d.sum()-missing)/len(records)),float((d.sum()+missing)/len(records))] if records else [-1,1])
    return result


def audit_saved(output):
    """No fits: verify saved data/anchors/df/gates and recompute descriptive summary."""
    plan=json.loads(PLAN_PATH.read_text()); records=[]
    for path in sorted(output.glob('rep_*.json')):
        r=json.loads(path.read_text());csv=output/f"data_{r['seed']}.csv"
        if digest(csv.read_bytes())!=r['data_sha256'] or r['plan_sha256']!=digest(PLAN_PATH.read_bytes()):
            raise ValueError('Saved data/plan hash mismatch')
        data=pd.read_csv(csv);g=data.pop('group');x=data.to_numpy()
        if list(data.columns)!=r['items'] or not np.isin(x,range(plan['categories'])).all():
            raise ValueError('Category/item alignment mismatch')
        if (int((g==0).sum()),int((g==1).sum()))!=(plan['n_reference'],plan['n_focal']):
            raise ValueError('Group counts differ')
        grm=r['methods']['GRM_LR']
        if grm['valid']:
            for item, hypotheses in grm['fits'].items():
                for f in hypotheses.values():
                    if not (f['converged'] and f['quadrature_ok'] and f['information_available'] and not f['at_bound']):
                        raise ValueError('Invalid GRM inference was released')
                    for anchor in plan['anchors']:
                        j=r['items'].index(anchor)
                        if f['a'][0][j]!=f['a'][1][j] or f['b'][0][j]!=f['b'][1][j]:
                            raise ValueError('Fixed-anchor equality violated')
            if any(row['sd (genel)']!=plan['categories'] for row in grm['table']):
                raise ValueError('GRM omnibus df mismatch')
        if 'reused_observed' in r:
            old=Path(r['reused_observed']['source']); raw=old.read_bytes()
            if digest(raw)!=r['reused_observed']['record_sha256']:
                raise ValueError('Original observed fit provenance changed')
            prior=json.loads(raw)
            if any(r['methods'][m]!=prior['methods'][m] for m in ('Mantel','ordinal_LR')):
                raise ValueError('Observed-score fits were not reused unchanged')
        records.append(r)
    if len({r['seed'] for r in records})!=len(records) or len({r['data_sha256'] for r in records})!=len(records):
        raise ValueError('Duplicated dataset/seed')
    prior=json.loads((output/'summary.json').read_text()); result=summarize(records,plan,precision_plan(plan),len(records)==precision_plan(plan)['required_repetitions'])
    result.update({k:v for k,v in prior.items() if k not in ('methods','paired','scientific_status')})
    result.update(saved_audit='PASSED: hashes, counts, categories, anchor equalities, df, inference gates, observed fit reuse',
                  reporting_code_sha256=digest(Path(__file__).read_bytes()),record_code_sha256=sorted({r['code_sha256'] for r in records}))
    assessment=dict(pilot_complete=len(records)>=plan['pilot_repetitions'],
        pilot_all_valid=all(v['valid'] for r in records for v in r['methods'].values()),
        projected_total_seconds=prior.get('estimated_required_seconds'),budget_seconds=plan['total_fit_budget_seconds'])
    reasons=[]
    if not assessment['pilot_complete']:reasons.append('INCOMPLETE PILOT')
    if not assessment['pilot_all_valid']:reasons.append('INVALID PILOT ANALYSES')
    if assessment['projected_total_seconds'] is None or assessment['projected_total_seconds']>assessment['budget_seconds']:
        reasons.append('PROJECTED COST EXCEEDS FIXED BUDGET')
    assessment['expansion_blockers']=reasons;result['expansion_assessment']=assessment
    save(output/'summary.json',result)
    return result


def run(output, reuse_observed=None):
    from validation.dif_validation_study import gen_ordinal
    from importlib.metadata import version
    plan=json.loads(PLAN_PATH.read_text()); output.mkdir(parents=True,exist_ok=True)
    plan_hash=digest(PLAN_PATH.read_bytes());code_hash=digest(Path(__file__).read_bytes())
    precision=precision_plan(plan);save(output/'precision_plan.json',precision)
    save(output/'mechanism.json',{f'{n}_{mu}':score_mechanism(n,mu) for n in (61,81) for mu in (0.,-.5)})
    records=[];total_seconds=0.;limit=plan['pilot_repetitions'];stop='pilot complete'
    for index in range(precision['required_repetitions']):
        if index>=limit:
            elapsed=sum(r['seconds'] for r in records)
            valid=all(v['valid'] for r in records for v in r['methods'].values())
            projection=elapsed/len(records)*(precision['required_repetitions']-len(records))
            if not valid or elapsed+projection>plan['total_fit_budget_seconds']:
                stop='NO EXPANSION: invalid pilot or projected precision-planned cost exceeds fixed budget';break
            limit=precision['required_repetitions']
        seed=plan['first_seed']+index; path=output/f'rep_{seed}.json';csv=output/f'data_{seed}.csv'
        if path.exists():
            r=json.loads(path.read_text())
            if r['plan_sha256']!=plan_hash or r['code_sha256']!=code_hash or digest(csv.read_bytes())!=r['data_sha256']:
                raise ValueError('Cached provenance differs; no silent rerun/overwrite')
            records.append(r);total_seconds+=r['seconds'];continue
        if total_seconds+plan['grm_timeout_seconds']>plan['total_fit_budget_seconds']:
            stop='BUDGET STOP before next dataset';break
        t=time.monotonic();prior=None
        if reuse_observed is not None:
            old_record=reuse_observed/f'rep_{seed}.json';old_data=reuse_observed/f'data_{seed}.csv'
            prior=json.loads(old_record.read_text())
            if prior['plan_sha256']!=plan_hash or digest(old_data.read_bytes())!=prior['data_sha256']:
                raise ValueError('Observed reuse provenance mismatch')
            csv.write_bytes(old_data.read_bytes());x=pd.read_csv(csv);g=x.pop('group')
        else:
            x,g=gen_ordinal(plan['n_reference'],plan['n_focal'],plan['theta_focal'][0],plan['true_dif'],seed)
            frame=x.copy();frame['group']=np.asarray(g);frame.to_csv(csv,index=False)
        r=dict(seed=seed,plan_sha256=plan_hash,code_sha256=code_hash,data_sha256=digest(csv.read_bytes()),
               items=list(x.columns),tested_family=plan['tested_family'],anchors=plan['anchors'],methods={})
        for method in ('Mantel','ordinal_LR'):
            r['methods'][method]=prior['methods'][method] if prior else observed(x,g,plan,method)
        if prior:
            r['reused_observed']=dict(source=str(old_record),record_sha256=digest(old_record.read_bytes()),
                                      code_sha256=prior['code_sha256'])
        worker=output/f'grm_{seed}.json'
        with (output/f'grm_{seed}.stdout').open('w') as stdout,(output/f'grm_{seed}.stderr').open('w') as stderr:
            try:
                subprocess.run([sys.executable,'-m','validation.ordinal_dif_paired.experiment','--grm-worker',str(csv),str(worker)],
                    cwd=ROOT,stdout=stdout,stderr=stderr,timeout=plan['grm_timeout_seconds'],check=True)
                r['methods']['GRM_LR']=json.loads(worker.read_text())
            except Exception as exc:r['methods']['GRM_LR']=dict(valid=False,reason=f'{type(exc).__name__}: {exc}')
        r['seconds']=time.monotonic()-t+(prior['seconds'] if prior else 0)
        total_seconds+=r['seconds'];save(path,r);records.append(clean(r))
        print(json.dumps(dict(seed=seed,valid={k:v['valid'] for k,v in r['methods'].items()},seconds=r['seconds'])),flush=True)
    result=summarize(records,plan,precision,len(records)==precision['required_repetitions'])
    result.update(stop_reason=stop,seconds=total_seconds,plan_sha256=plan_hash,code_sha256=code_hash,
                  versions={p:version(p) for p in ('numpy','scipy','pandas','statsmodels')},
                  estimated_required_seconds=(total_seconds/len(records)*precision['required_repetitions']) if records else None)
    save(output/'summary.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path);parser.add_argument('--grm-worker',nargs=2,type=Path)
    parser.add_argument('--reuse-observed',type=Path)
    parser.add_argument('--summarize',type=Path)
    args=parser.parse_args()
    if args.grm_worker:grm_worker(*args.grm_worker)
    elif args.summarize:audit_saved(args.summarize)
    elif args.run:run(args.run,args.reuse_observed)
    else:print(json.dumps(precision_plan(json.loads(PLAN_PATH.read_text())),indent=2))
