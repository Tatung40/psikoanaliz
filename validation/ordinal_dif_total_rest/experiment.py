"""Opt-in fixed-design paired experiment. Never executed by routine CI."""
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import signal
import subprocess
import time

import numpy as np
from scipy.stats import binomtest
from core.dif_classical import logistic_dif
from validation.dif_validation_study import gen_ordinal
from validation.ordinal_dif_paired.experiment import binomial_ci, classify, clean

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def save(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')


def paired_stats(total, rest):
    x, y = np.asarray(total, bool), np.asarray(rest, bool)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError('Equal one-dimensional complete paired outcomes required')
    n = len(x)
    if not n:
        return dict(n=0, delta=None, mcse=None, mcnemar_p=None,
                    exact_ci=[-1., 1.], hoeffding_ci=[-1., 1.])
    b, c = int(np.sum(x & ~y)), int(np.sum(~x & y))
    m = b+c
    delta = (b-c)/n
    pi = binomial_ci(m, n, .025)
    q = binomial_ci(b, m, .025)
    corners = [p*(2*v-1) for p in pi for v in q]
    margin = math.sqrt(2*math.log(40)/n)
    return dict(n=n, a=int(np.sum(x & y)), b=b, c=c,
                d=int(np.sum(~x & ~y)), delta=delta,
                mcse=math.sqrt(max(0., (m-(b-c)**2/n)/(n-1)/n)) if n>1 else None,
                mcnemar_p=float(binomtest(b,m,.5).pvalue) if m else 1.,
                no_discordances=not bool(m), pi_ci=pi, q_ci=q,
                exact_ci=[min(corners), max(corners)],
                hoeffding_ci=[max(-1.,delta-margin), min(1.,delta+margin)])


def summarize(records, planned=2500):
    n = len(records)
    complete = [r for r in records if all(r['methods'].get(k, {}).get('valid', False)
                                         for k in ('total', 'rest'))]
    confirm = n == planned and len(complete) == planned
    out = dict(attempted=n, valid_pairs=len(complete), failures=n-len(complete),
               fixed_design_complete=confirm, methods={})
    for name in ('total', 'rest'):
        valid=[r['methods'][name] for r in records if r['methods'].get(name,{}).get('valid',False)]
        k=len(valid); s=sum(r['holm_event'] for r in valid); bh=sum(r['bh_event'] for r in valid)
        raw=[z for r in valid for z in r['raw_rejections']]
        ci=binomial_ci(s,k,.025)
        out['methods'][name]=dict(valid=k, failed=n-k, holm_events=s,
            fwer=s/k if k else None, fwer_ci=ci,
            fwer_mcse=math.sqrt((s/k)*(1-s/k)/(k-1)) if k>1 else None,
            bh_global_null_fdr=bh/k if k else None,
            bh_ci=binomial_ci(bh,k,.025),
            raw_type1=float(np.mean(raw)) if raw else None,
            raw_type1_cluster_mcse=float(np.std([np.mean(r['raw_rejections']) for r in valid],ddof=1)/math.sqrt(k)) if k>1 else None,
            unknown_failure_empirical_bounds=[s/n,(s+n-k)/n] if n else [0,1],
            failure_robust_ci=[binomial_ci(s,n,.025)[0],binomial_ci(s+n-k,n,.025)[1]],
            decision=('SCOPED INFLATION' if ci[0]>.05 else 'SCOPED CONTROL' if ci[1]<=.05 else 'INCONCLUSIVE') if confirm else 'INCONCLUSIVE / INCOMPLETE')
    out['paired']=paired_stats([r['methods']['total']['holm_event'] for r in complete],
                               [r['methods']['rest']['holm_event'] for r in complete])
    p=out['paired']; missing=n-len(complete); diff=(p.get('b',0)-p.get('c',0))
    p['unknown_failure_empirical_bounds']=[(diff-missing)/n,(diff+missing)/n] if n else [-1,1]
    ci=p['hoeffding_ci']
    out['primary_decision']=('PAIRED DIRECTION DETECTED' if ci[0]>0 or ci[1]<0 else 'INCONCLUSIVE') if confirm else 'INCONCLUSIVE / INCOMPLETE'
    out['warning']='McNemar tests equal marginal events; it does not certify nominal FWER. BH event=FDR only under this global null. No general certification.'
    return out


def analyze(data, group, method):
    start=time.monotonic(); total=data.sum(axis=1).to_numpy(); raw=[]
    try:
        for name in ('P09','P10'):
            y=data[name].to_numpy()
            raw.append(logistic_dif(y,group,total-y if method=='rest' else total,ordinal=True))
        result=classify([r['p_omnibus'] for r in raw],.05)
    except Exception as exc:
        result=dict(valid=False,reason=f'{type(exc).__name__}: {exc}')
    result.update(raw=raw,seconds=time.monotonic()-start)
    return result


class BudgetExpired(BaseException):
    pass


def execute():
    plan=json.loads((HERE/'plan.json').read_text())
    output=HERE/'results'
    output.mkdir(exist_ok=True)
    stream=output/'records.jsonl'
    if stream.exists() or (output/'records.jsonl.gz').exists():
        raise RuntimeError('Existing evidence protected: no overwrite/refit')
    source_paths=['validation/ordinal_dif_total_rest/experiment.py',
                  'validation/ordinal_dif_total_rest/plan.json',
                  'validation/dif_validation_study.py','core/dif_classical.py',
                  'validation/ordinal_dif_paired/experiment.py']
    save(output/'metadata.json',dict(python=platform.python_version(),
        versions={p:importlib.metadata.version(p) for p in ('numpy','scipy','pandas','statsmodels')},
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths},
        plan=plan, clock='monotonic wall seconds', engine='production ordinal LR only'))
    records=[]; start=time.monotonic(); stop='fixed design completed'; pilot=None
    def deadline(*_):
        raise BudgetExpired()
    old=signal.signal(signal.SIGALRM,deadline)
    signal.setitimer(signal.ITIMER_REAL,plan['budget_seconds'])
    current=None; data=None; group=None
    try:
        with stream.open('x') as f:
            for i in range(plan['n_required']):
                t=time.monotonic(); seed=2026110100+i
                data,group=gen_ordinal(500,500,-.5,False,seed)
                payload=data.to_numpy(dtype=np.uint8).tobytes()+group.astype(np.uint8).tobytes()
                current=dict(index=i,seed=seed,shape=list(data.shape),columns=list(data.columns),
                    data_sha256=hashlib.sha256(payload).hexdigest(),methods={})
                try:
                    for method in ('total','rest'):
                        current['methods'][method]=analyze(data,group,method)
                except BudgetExpired:
                    current['interrupted']='1800s budget deadline'
                    stop='budget deadline'
                current['seconds']=time.monotonic()-t
                valid=all(current['methods'].get(k,{}).get('valid',False) for k in ('total','rest'))
                if i<30 or not valid:
                    path=output/f'data_{i:04d}.npz'
                    np.savez_compressed(path,X=data.to_numpy(dtype=np.uint8),group=group.astype(np.uint8))
                    current['matrix_file']=path.name
                    current['matrix_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
                records.append(current)
                f.write(json.dumps(clean(current),allow_nan=False)+'\n'); f.flush()
                current=None
                if stop=='budget deadline': break
                if len(records)==30:
                    elapsed=time.monotonic()-start
                    valid_count=sum(all(r['methods'].get(k,{}).get('valid',False) for k in ('total','rest')) for r in records)
                    pilot=dict(n=30,valid_pairs=valid_count,seconds=elapsed,
                               projected_total_seconds=elapsed/30*plan['n_required'])
                    pilot['continue']=valid_count==30 and pilot['projected_total_seconds']<=plan['budget_seconds']
                    save(output/'feasibility.json',pilot)
                    print(json.dumps(pilot),flush=True)
                    if not pilot['continue']:
                        stop='prespecified feasibility stop'; break
                elif len(records)%100==0:
                    print(f'Completed {len(records)} pairs; {time.monotonic()-start:.1f}s',flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,old)
    elapsed=time.monotonic()-start
    raw=stream.read_bytes(); packed=gzip.compress(raw,mtime=0)
    assert gzip.decompress(packed)==raw
    (output/'records.jsonl.gz').write_bytes(packed); stream.unlink()
    report=summarize(records,plan['n_required'])
    report.update(seconds=elapsed,stop=stop,feasibility=pilot,
                  records_sha256=hashlib.sha256(raw).hexdigest(),
                  records_gzip_sha256=hashlib.sha256(packed).hexdigest())
    save(output/'summary.json',report)
    print(json.dumps(clean(report),indent=2),flush=True)


if __name__=='__main__':
    execute()
