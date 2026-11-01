"""Exactly ten impact-zero datasets; reuse all negative-impact fits unchanged."""
import argparse
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.special import expit

from validation.dif_validation_study import gen_ordinal
from validation.ordinal_dif_paired import experiment as E

ROOT=E.ROOT
HERE=Path(__file__).parent
OLD=HERE/'evidence'
CONTROL_PLAN=HERE/'control_plan.json'
_DEFAULT_RNG=np.random.default_rng


class TraceRNG:
    def __init__(self,seed):self.rng=_DEFAULT_RNG(seed);self.calls=[];self.values=[]
    def capture(self,method,*args,**kwargs):
        value=getattr(self.rng,method)(*args,**kwargs)
        self.calls.append(dict(method=method,shape=list(value.shape),sha256=E.digest(value.tobytes())))
        self.values.append(value.copy());return value
    def standard_normal(self,*a,**kw):return self.capture('standard_normal',*a,**kw)
    def random(self,*a,**kw):return self.capture('random',*a,**kw)


def control_data(seed,old_data,nr=500,nf=500):
    """Trace the actual control calls, reconstruct old responses, fit nothing old."""
    trace=TraceRNG(seed)
    with patch('numpy.random.default_rng',return_value=trace):x,g=gen_ordinal(nr,nf,0.,False,seed)
    expected=[('standard_normal',[nr]),('standard_normal',[nf])]+[('random',[nr+nf])]*10
    if [(c['method'],c['shape']) for c in trace.calls]!=expected:
        raise ValueError('Generator RNG consumption changed; seed equality alone is insufficient')
    theta=np.r_[trace.values[0],-.5+trace.values[1]]
    a=np.linspace(.9,1.8,10);loc=np.linspace(-1,1,10);minus=np.zeros_like(x.to_numpy())
    for j in range(10):
        thresholds=loc[j]+np.array([-1.,0.,1.])
        surv=expit(a[j]*(theta[:,None]-thresholds[None,:]))
        minus[:,j]=(trace.values[j+2][:,None]<surv).sum(1)
    if not np.array_equal(minus,old_data.to_numpy()):raise ValueError('Shared draws do not reconstruct saved negative-impact matrix')
    if not np.array_equal(x.iloc[:nr],old_data.iloc[:nr]):raise ValueError('Reference rows differ')
    if not np.all(x.iloc[nr:].to_numpy()>=minus[nr:]):raise ValueError('Focal common-uniform monotonicity failed')
    proof=dict(bit_generator=type(trace.rng.bit_generator).__name__,calls=trace.calls,
               reconstructed_negative_exactly=True,identical_reference_rows=True,focal_monotone=True,
               interpretation='Verified common latent standard normals and item uniforms; outcome correlation/variance reduction is not assumed')
    return x,g,proof


def pair_contrast(a,b,total):
    """a-b for complete dataset pairs; bounded outcomes, marginal 95% interval."""
    d=np.asarray(a,float)-np.asarray(b,float);n=len(d);missing=total-n
    eps=math.sqrt(2*math.log(2/.05)/n) if n else 2
    corr=float(np.corrcoef(a,b)[0,1]) if n>1 and np.std(a)>0 and np.std(b)>0 else None
    return dict(complete_pairs=n,missing_pairs=missing,differences=d.tolist(),
        mean=float(d.mean()) if n else None,mcse=float(d.std(ddof=1)/math.sqrt(n)) if n>1 else None,
        marginal_ci95=[max(-1.,float(d.mean())-eps),min(1.,float(d.mean())+eps)] if n else [-1.,1.],
        missing_pair_range=[float((d.sum()-missing)/total),float((d.sum()+missing)/total)] if total else [-1.,1.],
        a_greater=int(np.sum(d>0)),b_greater=int(np.sum(d<0)),
        empirical_outcome_correlation=corr,interpretation='Diagnostic paired contrast; marginal conservative interval, not joint certification')


def metric(row,name):
    return float(np.mean(row['raw_rejections'])) if name=='raw_type1' else int(row[name])


def analyze(output):
    base=json.loads(E.PLAN_PATH.read_text());control=json.loads(CONTROL_PLAN.read_text())
    records=[json.loads(p.read_text()) for p in sorted(output.glob('rep_*.json'))]
    old={r['seed']:r for r in (json.loads(p.read_text()) for p in sorted(OLD.glob('rep_*.json')))}
    if [r['seed'] for r in records]!=control['seeds']:raise ValueError('Incomplete/changed ten-seed family')
    for r in records:
        if E.digest((output/f"data_{r['seed']}.csv").read_bytes())!=r['data_sha256']:raise ValueError('Control data hash changed')
        if E.digest((OLD/f"rep_{r['seed']}.json").read_bytes())!=r['minus_record_sha256']:raise ValueError('Cached negative fit changed')
        if not r['crn']['reconstructed_negative_exactly']:raise ValueError('No verified common random numbers')
    precision=json.loads((OLD/'precision_plan.json').read_text())
    summary=E.summarize(records,base,precision,False)
    summary.update(impact=0.,control_plan_sha256=E.digest(CONTROL_PLAN.read_bytes()),
        seconds=sum(r['seconds'] for r in records),scientific_status='CONTROL PILOT / INCONCLUSIVE / NOT CERTIFIED')
    E.save(output/'summary.json',summary)
    comparisons={};within={}
    for method in base['methods']:
        valid=[r for r in records if r['methods'][method]['valid'] and old[r['seed']]['methods'][method]['valid']]
        comparisons[method]={name:pair_contrast([metric(r['methods'][method],name) for r in valid],
            [metric(old[r['seed']]['methods'][method],name) for r in valid],len(records)) for name in ('holm_event','bh_event','raw_type1')}
        for row in comparisons[method].values():row['paired_seeds']=[r['seed'] for r in valid]
    for condition,rs in (('impact0',records),('impact_minus0.5',list(old.values()))):
        within[condition]={}
        for a,b in itertools.combinations(base['methods'],2):
            valid=[r for r in rs if r['methods'][a]['valid'] and r['methods'][b]['valid']]
            within[condition][a+' minus '+b]={name:pair_contrast([metric(r['methods'][a],name) for r in valid],
                [metric(r['methods'][b],name) for r in valid],len(rs)) for name in ('holm_event','bh_event','raw_type1')}
            for row in within[condition][a+' minus '+b].values():row['paired_seeds']=[r['seed'] for r in valid]
    result=dict(control_summary=summary,minus_summary_sha256=E.digest((OLD/'summary.json').read_bytes()),
        minus_methods_reused=json.loads((OLD/'summary.json').read_text())['methods'],
        impact0_minus_impact_minus0_5=comparisons,within_condition=within,
        crn_verified_datasets=len(records),no_negative_condition_refits=True,
        reporting_code_sha256=E.digest(Path(__file__).read_bytes()),
        confidence_note='Primary method intervals: joint >=95% across three methods WITHIN each condition only. Contrasts and BH intervals are marginal 95%; no joint claim across six cells or contrasts.')
    E.save(output/'comparison.json',result);return result


def run(output):
    from importlib.metadata import version
    plan=json.loads(CONTROL_PLAN.read_text());base=json.loads(E.PLAN_PATH.read_text())
    if E.digest(E.PLAN_PATH.read_bytes())!=plan['base_plan_sha256']:raise ValueError('Original design changed')
    minus_summary=json.loads((OLD/'summary.json').read_text())
    if minus_summary['versions']['numpy']!=np.__version__:raise ValueError('NumPy version changed; cannot claim original RNG equivalence')
    generator=ROOT/'validation/dif_validation_study.py'
    source_bytes=subprocess.check_output(['git','show',plan['base_commit']+':validation/dif_validation_study.py'],cwd=ROOT)
    if source_bytes!=generator.read_bytes():raise ValueError('Original generator source changed')
    output.mkdir(parents=True,exist_ok=True);elapsed=0.
    for seed in plan['seeds']:
        path=output/f'rep_{seed}.json';csv=output/f'data_{seed}.csv'
        if path.exists():
            r=json.loads(path.read_text())
            if r['control_plan_sha256']!=E.digest(CONTROL_PLAN.read_bytes()) or r['code_sha256']!=E.digest(Path(__file__).read_bytes()):
                raise ValueError('Changed cache; no silent rerun')
            elapsed+=r['seconds'];continue
        if elapsed+plan['grm_timeout_seconds']>plan['budget_seconds']:raise RuntimeError('Control budget stop; retain partial evidence')
        t=time.monotonic();minus_path=OLD/f'rep_{seed}.json';minus=json.loads(minus_path.read_text())
        old_csv=OLD/f'data_{seed}.csv'
        if E.digest(old_csv.read_bytes())!=minus['data_sha256'] or minus['plan_sha256']!=plan['base_plan_sha256']:
            raise ValueError('Original negative evidence changed')
        old=pd.read_csv(old_csv);old_group=old.pop('group')
        x,g,proof=control_data(seed,old)
        if not np.array_equal(g,old_group):raise ValueError('Group/row alignment changed')
        frame=x.copy();frame['group']=g;frame.to_csv(csv,index=False)
        r=dict(seed=seed,impact=0.,control_plan_sha256=E.digest(CONTROL_PLAN.read_bytes()),base_plan_sha256=plan['base_plan_sha256'],
            code_sha256=E.digest(Path(__file__).read_bytes()),worker_code_sha256=E.digest(Path(E.__file__).read_bytes()),
            generator_sha256=E.digest(source_bytes),data_sha256=E.digest(csv.read_bytes()),
            minus_record_sha256=E.digest(minus_path.read_bytes()),minus_data_sha256=minus['data_sha256'],crn=proof,
            items=list(x.columns),anchors=base['anchors'],tested_family=base['tested_family'],
            versions={p:version(p) for p in ('numpy','scipy','pandas','statsmodels')},methods={})
        for method in ('Mantel','ordinal_LR'):r['methods'][method]=E.observed(x,g,base,method)
        worker=output/f'grm_{seed}.json';logs=[output/f'grm_{seed}.{suffix}' for suffix in ('stdout','stderr')]
        with logs[0].open('w') as stdout,logs[1].open('w') as stderr:
            try:
                subprocess.run([sys.executable,'-m','validation.ordinal_dif_paired.experiment','--grm-worker',str(csv),str(worker)],
                    cwd=ROOT,stdout=stdout,stderr=stderr,timeout=plan['grm_timeout_seconds'],check=True)
                r['methods']['GRM_LR']=json.loads(worker.read_text())
            except Exception as exc:r['methods']['GRM_LR']=dict(valid=False,reason=f'{type(exc).__name__}: {exc}')
        r['worker_logs']={p.suffix[1:]:dict(bytes=p.stat().st_size,sha256=E.digest(p.read_bytes())) for p in logs}
        r['seconds']=time.monotonic()-t;elapsed+=r['seconds'];E.save(path,r)
        if worker.exists() and json.loads(worker.read_text())==E.clean(r['methods']['GRM_LR']):worker.unlink()
        for p in logs:
            if p.stat().st_size==0:p.unlink()
        print(json.dumps(dict(seed=seed,valid={m:v['valid'] for m,v in r['methods'].items()},seconds=r['seconds'])),flush=True)
    return analyze(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path);parser.add_argument('--summarize',type=Path)
    args=parser.parse_args()
    if args.run:run(args.run)
    elif args.summarize:analyze(args.summarize)
