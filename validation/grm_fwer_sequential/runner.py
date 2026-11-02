"""Opt-in, budgeted fixed-look experiment. No automatic CI calibrations."""
import concurrent.futures
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.stats import chi2
from core import grm_dif as G
from validation.gh81_diagnostic import diagnostic as D
from validation.gh_reoptimization import experiment as E
from validation.grm_gh101_stage_a.runner import dense_information
from validation.grm_gh101_uniform_pilot.pilot import layouts, cold_start
from validation.ordinal_dif_paired.control import TraceRNG
from validation.dif_validation_study import gen_ordinal
from validation.grm_gh101_stage_a.audit import adjusted
from validation.grm_fwer_sequential.statistics import LOOKS, summary

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCES=('core/grm_dif.py','validation/gh81_diagnostic/diagnostic.py',
    'validation/gh_reoptimization/experiment.py','validation/grm_gh101_stage_a/runner.py',
    'validation/grm_gh101_uniform_pilot/pilot.py','validation/dif_validation_study.py',
    'validation/ordinal_dif_paired/control.py','validation/grm_fwer_sequential/runner.py',
    'validation/grm_fwer_sequential/statistics.py','validation/grm_fwer_sequential/protocol.json')


def generate(seed, impact, expected):
    trace=TraceRNG(seed)
    with patch('numpy.random.default_rng',return_value=trace):
        frame,g=gen_ordinal(500,500,-.5 if impact=='minus' else 0.,False,seed)
    x=frame.to_numpy(dtype=np.uint8);g=g.astype(np.uint8)
    digest=D.sha(x.tobytes()+g.tobytes())
    if digest != expected:raise ValueError('Archived response hash mismatch')
    if [(v['method'],v['shape']) for v in trace.calls] != [('standard_normal',[500])]*2+[('random',[1000])]*10:
        raise ValueError('RNG consumption mismatch')
    return x,g,trace.calls


def run_case(seed, impact, expected, output):
    start=time.monotonic();result=dict(seed=seed,impact=impact,valid=False,solutions={},lr=[],data_sha256=expected)
    def alarm(*args):raise TimeoutError('Registered120s per dataset')
    previous=signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,120)
    try:
        x,g,draws=generate(seed,impact,expected);result['rng_draws']=draws
        if seed<2026110130:np.savez_compressed(output/f'{seed}_{impact}.npz',X=x,group=g)
        columns=[f'P{i:02d}' for i in range(1,11)]
        inputs=G.validate_inputs(pd.DataFrame(x,columns=columns),g,0,1,columns[:8],categories=4,min_cell=5)
        fitter=G.Fitter(inputs,points=101);fitter.audit_grid=D.reference_grid(161)
        designs,slots=layouts();wanted={slots[item+'.'+h] for item in ('P09','P10') for h in ('H0','H2')}
        result['slots']={k:v for k,v in slots.items() if k.endswith(('.H0','.H2'))}
        for key in designs:
            if key not in wanted:continue
            d=designs[key]['design'];expected_start,_=cold_start(d,inputs)
            original=G.minimize;calls=[]
            def tracked(fn,x0,*args,**kwargs):
                np.testing.assert_array_equal(x0,expected_start)
                calls.append(dict(method=kwargs['method'],options=kwargs['options'],x0_sha256=D.sha(np.asarray(x0).tobytes())))
                return original(fn,x0,*args,**kwargs)
            with patch.object(G,'minimize',tracked):f=fitter.fit(d.roles,start_fit=None,information=True,max_iter=2000)
            a,b,mu,sd=d.unpack(f['params'])
            s={k:f[k] for k in ('params','converged','at_bound','score','audit_score','quadrature_ok','quadrature_error','information_min_eigenvalue','information_condition','iterations','reason')}
            s.update(roles=d.roles,optimizer=calls,anchor_equality=E.anchor_feasible(f),covariance_available=f['cov'] is not None)
            result['solutions'][key]=s
            if len(calls)!=1 or not (f['converged'] and not f['at_bound'] and f['quadrature_ok'] and f['cov'] is not None and s['anchor_equality']):
                raise ArithmeticError('Native convergence/bounds/information/GH161/anchor gate')
            s['evaluations']={}
            for q,grid in [(101,fitter.grid),(161,fitter.audit_grid)]:
                value,gradient=G.objective(f['params'],d,fitter.parts,grid)
                independent=-len(x)*D.reference_nll(f['params'],d,x,g,q)
                s['evaluations'][str(q)]=dict(loglik=-len(x)*value,gradient=gradient,independent_loglik=independent)
                if abs(independent+len(x)*value)>1e-8:raise ArithmeticError('Independent logLik mismatch')
            dense=dense_information(f['params'],d,fitter.parts,fitter.audit_grid,len(x));s['dense_information']=dense
            if not dense['valid']:raise ArithmeticError('GH161 dense information')
            s['derivatives']={}
            for h in (1e-4,1e-5):
                fd,_=D.finite_difference(f['params'],d,x,g,161,h)
                np.testing.assert_allclose(fd,s['evaluations']['161']['gradient'],atol=1e-8,rtol=1e-5)
                s['derivatives'][str(h)]=fd
        if fitter.optimizations!=3:raise ArithmeticError('Omnibus requires exactly3 unique fits')
        p=[]
        for item in ('P09','P10'):
            null,alt=[result['solutions'][slots[item+'.'+h]] for h in ('H0','H2')]
            lr={q:2*(alt['evaluations'][q]['loglik']-null['evaluations'][q]['loglik']) for q in ('101','161')}
            df=G.Design(alt['roles'],4).size-G.Design(null['roles'],4).size
            if df!=4 or min(lr.values())< -2e-6 or abs(lr['101']-lr['161'])>.004:raise ArithmeticError('Nested LR/df/density gate')
            p.append(float(chi2.sf(max(0,lr['161']),4)));result['lr'].append(dict(item=item,df=df,lr=lr,p=p[-1]))
        result['valid']=True
        result['decisions']=dict(p=p,holm=adjusted(p,'holm'),bh=adjusted(p,'bh'),
            holm_event=any(v<.05 for v in adjusted(p,'holm')),bh_event=any(v<.05 for v in adjusted(p,'bh')))
    except Exception as exc:
        result['failure']=f'{type(exc).__name__}: {exc}'
        result['fatal']=isinstance(exc,ValueError) and ('hash' in str(exc) or 'RNG' in str(exc))
        result['decisions']=dict(p=None,holm_event=None,bh_event=None)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
    result.update(seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    return E.clean(result)


def ordinal_hashes():
    hashes={};archives={}
    for impact,folder in [('minus','ordinal_dif_total_rest'),('zero','ordinal_dif_impact_zero')]:
        path=ROOT/'validation'/folder/'results/records.jsonl.gz';archives[str(path.relative_to(ROOT))]=D.sha(path.read_bytes())
        for line in gzip.decompress(path.read_bytes()).splitlines():
            row=json.loads(line);hashes[row['seed'],impact]=row['data_sha256']
    return hashes,archives


def run():
    plan=json.loads((HERE/'protocol.json').read_text());output=HERE/'results'
    if output.exists():raise RuntimeError('Evidence exists; no automatic rerun')
    versions={k:importlib.metadata.version(k) for k in ('numpy','scipy','pandas')}
    if any(versions[k]!=plan['environment'][k] for k in versions):raise ValueError('Registered environment mismatch')
    hashes,archives=ordinal_hashes();sources={p:D.sha((ROOT/p).read_bytes()) for p in SOURCES}
    output.mkdir();start=time.monotonic();rows=[];stop='maximum fixed look reached'
    E.save(output/'metadata.json',dict(protocol_commit='9d01c0fdd8681e85e43437906459752353dc003d',sources=sources,archives=archives,versions=versions,execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),historical_fits_reused=0))
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
            for look in LOOKS:
                first=2026110100+len(rows)//2
                for seed in range(first,2026110100+look):
                    if time.monotonic()-start>plan['resources']['total_wall_seconds']-120:stop='registered wall budget';break
                    futures=[pool.submit(run_case,seed,impact,hashes[seed,impact],output) for impact in ('minus','zero')]
                    pair=[f.result() for f in futures]
                    if pair[0].get('rng_draws')!=pair[1].get('rng_draws'):raise ValueError('Paired random draws mismatch')
                    for r in pair:
                        with gzip.open(output/'records.jsonl.gz','ab') as stream:stream.write((json.dumps(r,allow_nan=False)+'\n').encode())
                        rows.append(r)
                    E.save(output/'progress.json',dict(pairs=len(rows)//2,seconds=time.monotonic()-start,valid=sum(r['valid'] for r in rows)))
                    print(f'{len(rows)//2} pairs; valid={sum(r["valid"] for r in rows)}/{len(rows)}; {time.monotonic()-start:.1f}s',flush=True)
                    if any(r.get('fatal') or r['peak_rss_mib']>1024 for r in pair):stop='technical integrity/memory gate';break
                if len(rows)//2 !=look:break
                current=summary(rows);E.save(output/f'look_{look:04d}.json',current)
                if current['stop_at_look']:stop='registered classification boundary';break
    finally:
        if {p:D.sha((ROOT/p).read_bytes()) for p in SOURCES}!=sources:raise ValueError('Source changed during execution')
        current=summary(rows);current.update(stop=stop,seconds=time.monotonic()-start,certified=False)
        E.save(output/'summary.json',current)
        E.save(output/'SHA256.json',{p.name:D.sha(p.read_bytes()) for p in output.iterdir() if p.is_file() and p.name!='SHA256.json'})

if __name__=='__main__':run()
