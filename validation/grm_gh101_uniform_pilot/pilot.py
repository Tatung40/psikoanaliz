"""Twenty archived datasets, uniform per-hypothesis cold GH101; no MC pooling."""
import gzip
import importlib.metadata
import json
from pathlib import Path
import resource
import signal
import subprocess
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
from core import grm_dif as G
from validation.gh_reoptimization import experiment as E
from validation.gh81_diagnostic import diagnostic as D

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
LEGACY=ROOT/'validation/ordinal_dif_grm_lr_paired/results'
SEEDS=tuple(range(2026110100,2026110110))


def protect():
    m=json.loads((HERE/'prior_manifest.json').read_text())
    for p,h in m.items():
        if D.sha((ROOT/p).read_bytes())!=h:raise ValueError('Protected source/evidence changed: '+p)
    return len(m)


def legacy_records():
    rows=json.loads(gzip.decompress((LEGACY/'records.json.gz').read_bytes()))['records']
    if len(rows)!=20 or {(r['seed'],r['impact']) for r in rows}!={(s,k) for s in SEEDS for k in ('minus','zero')}:raise ValueError('Preselected archive seed family differs')
    return {(r['seed'],r['impact']):r for r in rows}


def load_matrix(r):
    path=LEGACY/r['matrix_file']
    if D.sha(path.read_bytes())!=r['matrix_sha256']:raise ValueError('Matrix archive hash mismatch')
    with np.load(path) as data:x,g=data['X'].copy(),data['group'].copy()
    if D.sha(x.tobytes()+g.tobytes())!=r['data_sha256']:raise ValueError('Response/group content hash mismatch')
    if x.shape!=(1000,10) or np.bincount(g).tolist()!=[500,500]:raise ValueError('Data layout/group size mismatch')
    return x,g


def verify_pair(seed,legacy):
    minus,gm=load_matrix(legacy[(seed,'minus')]);zero,gz=load_matrix(legacy[(seed,'zero')])
    if not np.array_equal(gm,gz) or not np.array_equal(minus[:500],zero[:500]):raise ValueError('Reference/group pairing differs')
    if not np.all(zero[500:]>=minus[500:]):raise ValueError('Common-uniform focal monotonicity differs')
    trace=legacy[(seed,'minus')]['crn']
    if trace!=legacy[(seed,'zero')]['crn']:raise ValueError('Archived CRN traces differ')
    rng=np.random.default_rng(seed)
    replay=[('standard_normal',rng.standard_normal(500)),('standard_normal',rng.standard_normal(500))]
    replay += [('random',rng.random(1000)) for _ in range(10)]
    for (method,values),old in zip(replay,trace['draws']):
        if method!=old['method'] or list(values.shape)!=old['shape'] or D.sha(values.tobytes())!=old['sha256']:raise ValueError('Actual RNG draw replay differs')
    if len(trace['draws'])!=12:raise ValueError('RNG consumption count differs')
    return dict(seed=seed,rng_hashes_verified=12,reference_rows_identical=True,groups_identical=True,focal_monotone=True,
                source_crn_sha256=D.sha(json.dumps(trace,sort_keys=True).encode()),matrix_hashes={k:legacy[(seed,k)]['data_sha256'] for k in ('minus','zero')})


def layouts():
    designs={};slots={}
    for item in ('P09','P10'):
        for h,role in (('H0','anchor'),('Hb','b'),('Ha','a'),('H2','ab')):
            roles=['anchor']*8+['ab']*2;roles[int(item[1:])-1]=role
            key=D.sha(json.dumps(roles).encode());slots[item+'.'+h]=key
            if key not in designs:designs[key]=dict(design=G.Design(roles,4),first_slot=item+'.'+h)
    return designs,slots


def cold_start(design,inputs):
    x=inputs['x']
    tails=np.array([[(x[:,i]>=c).sum()/(x[:,i]>=0).sum() for c in range(1,design.categories)] for i in range(x.shape[1])])
    b=-np.log(np.clip(tails,.02,.98)/(1-np.clip(tails,.02,.98)))
    raw=design.encode(np.ones((2,len(design.roles))),np.stack([b,b]))
    lo,hi=np.array(design.bounds).T
    return np.clip(raw,lo,hi),bool(not np.array_equal(raw,np.clip(raw,lo,hi)))


def decision(records,stop):
    expected={(s,k) for s in SEEDS for k in ('minus','zero')}
    keys=[(r['seed'],r['impact']) for r in records]
    return 'PASS' if len(records)==20 and set(keys)==expected and len(set(keys))==20 and all(r.get('valid',False) for r in records) and stop=='fixed pilot complete' else 'FAIL'


class PilotStop(Exception):pass


def run_case(seed,impact,old,checkpoint):
    x,g=load_matrix(old);columns=[f'P{i:02d}' for i in range(1,11)]
    inputs=G.validate_inputs(pd.DataFrame(x,columns=columns),g,0,1,columns[:8],categories=4,min_cell=5)
    fitter=G.Fitter(inputs,points=101);fitter.audit_grid=D.reference_grid(161)
    designs,slots=layouts();result=dict(seed=seed,impact=impact,data_sha256=old['data_sha256'],slots=slots,solutions={},lr=[],valid=False,
        fits_attempted=0,fits_completed=0,optimizer_calls=[],initialization='native default, start_fit=None for every unique hypothesis')
    checkpoint(result)
    for key,m in designs.items():
        d=m['design'];expected,clipped=cold_start(d,inputs)
        result['active_model']=dict(solution=key,slot=m['first_slot']);result['fits_attempted']+=1;checkpoint(result)
        original=G.minimize
        def tracked(fun,x0,*args,**kwargs):
            call=dict(solution=key,x0=np.asarray(x0).copy(),expected_cold_sha256=D.sha(expected.tobytes()),cold_verified=bool(np.array_equal(x0,expected)),
                      native_initialization_clipped=clipped,method=kwargs.get('method'),options=kwargs.get('options'),jac=kwargs.get('jac'))
            result['optimizer_calls'].append(call);checkpoint(result)
            if not call['cold_verified']:raise PilotStop('Actual optimizer start is not native cold default')
            fitted=original(fun,x0,*args,**kwargs)
            call.update(scipy_success=bool(fitted.success),scipy_status=int(fitted.status),scipy_message=str(fitted.message),nfev=int(fitted.nfev),njev=int(fitted.njev))
            return fitted
        t=time.monotonic()
        with patch.object(G,'minimize',tracked):
            f=fitter.fit(d.roles,start_fit=None,information=True,max_iter=2000)
        result['fits_completed']+=1
        a,b,mu,sd=d.unpack(f['params'])
        s=dict(first_slot=m['first_slot'],roles=d.roles,params=f['params'].copy(),a=a,b=b,focal_mean=mu,focal_sd=sd,
               converged=f['converged'],at_bound=f['at_bound'],native_score=f['score'],audit_score=f['audit_score'],quadrature_ok=f['quadrature_ok'],quadrature_error=f['quadrature_error'],
               native_information_min_eigenvalue=f['information_min_eigenvalue'],native_information_condition=f['information_condition'],covariance_available=f['cov'] is not None,
               anchor_equality=E.anchor_feasible(f),reason=f['reason'],iterations=f['iterations'],fit_seconds=time.monotonic()-t,passed=False)
        result['solutions'][key]=s;checkpoint(result)
        native_ok=bool(f['converged'] and not f['at_bound'] and f['quadrature_ok'] and f['cov'] is not None and s['anchor_equality'])
        if not native_ok:result['failure_reason']='Native convergence/bounds/information/quadrature/anchor gate';return result
        evaluations={}
        for q,grid in ((101,fitter.grid),(161,fitter.audit_grid)):
            value,score=G.objective(f['params'],d,fitter.parts,grid);j=int(np.argmax(np.abs(score)))
            evaluations[str(q)]=dict(loglik=-len(x)*value,gradient=score,max_abs_normalized_gradient=float(abs(score[j])),max_parameter=D.labels(d)[j],
                independent_loglik=-len(x)*D.reference_nll(f['params'],d,x,g,q))
        dense=E.dense_information(f['params'],d,fitter.parts,fitter.audit_grid,len(x))
        derivative_checks={};derivatives_ok=True
        for h in (1e-4,1e-5):
            fd,_=D.finite_difference(f['params'],d,x,g,161,h);analytic=evaluations['161']['gradient']
            derivative_checks[str(h)]=dict(gradient=fd,max_abs_error=float(np.max(np.abs(fd-analytic))))
            derivatives_ok &= bool(np.allclose(fd,analytic,atol=1e-8,rtol=1e-5))
        slot=m['first_slot'];item,hyp=slot.split('.')
        old_fit=old['result']['fits'][item][hyp];od,op,_=D.recover(item,hyp,old_fit)
        old_value,old_score=G.objective(op,od,fitter.parts,fitter.audit_grid)
        oa,ob,om,osd=od.unpack(op)
        comparison=dict(old_native_GH61_loglik=old_fit['loglik'],old_fixed_GH161_loglik=-len(x)*old_value,old_fixed_GH161_score=float(np.max(np.abs(old_score))),
            new_minus_old_common_GH161_loglik=evaluations['161']['loglik']+len(x)*old_value,
            max_physical_parameter_change=float(max(np.max(np.abs(a-oa)),np.max(np.abs(b-ob)),abs(mu-om),abs(sd-osd))),
            interpretation='paired diagnostic only; legacy GH61 within-dataset warm hierarchy differs from this strict cold policy; no pooling or pure-grid causal claim')
        s.update(evaluations=evaluations,dense_information=dense,derivative_checks=derivative_checks,derivatives_ok=derivatives_ok,legacy_comparison=comparison,
                 passed=bool(dense['valid'] and derivatives_ok))
        checkpoint(result)
        if not s['passed']:result['failure_reason']='Dense information or independent derivative gate';return result
        result.pop('active_model',None)
    if len(result['optimizer_calls'])!=7 or fitter.optimizations!=7:result['failure_reason']='Unexpected cache/optimizer call count';return result
    table=E.lr_table(slots,result['solutions'],101,161)
    for row in table:
        null={'omnibus':'H0','thresholds':'Ha','slope':'Hb'}[row['test']];alt=result['solutions'][slots[row['item']+'.H2']];base=result['solutions'][slots[row['item']+'.'+null]]
        old61,note61=G._nested_lr(alt['legacy_comparison']['old_native_GH61_loglik'],base['legacy_comparison']['old_native_GH61_loglik'])
        old161,note161=G._nested_lr(alt['legacy_comparison']['old_fixed_GH161_loglik'],base['legacy_comparison']['old_fixed_GH161_loglik'])
        row.update(legacy_GH61_lr=old61,legacy_GH61_note=note61,legacy_common_GH161_lr=old161,legacy_common_GH161_note=note161,new_minus_legacy_common_GH161_lr=row['lr']['161']-old161)
    result['lr']=table;result['valid']=all(r['valid'] for r in table)
    if not result['valid']:result['failure_reason']='Native nested-LR gate'
    result['native_optimizer_calls']=fitter.optimizations
    return result


def summarize(records,elapsed,stop):
    valid=[r for r in records if r.get('valid',False)]
    nfit=sum(r.get('fits_attempted',0) for r in records);completed=sum(r.get('fits_completed',0) for r in records)
    models=[s for r in records for s in r.get('solutions',{}).values()]
    converged=sum(len(r.get('solutions',{}))==7 and all(s['converged'] for s in r['solutions'].values()) for r in records)
    audit_scores=[s['audit_score'] for s in models]
    return dict(technical_decision=decision(records,stop),scientific_classification='TECHNICAL PILOT ONLY / NOT CERTIFIED',stop=stop,planned=20,attempted=len(records),not_run=20-len(records),valid=len(valid),invalid=len(records)-len(valid),
        converged_datasets=converged,numerical_validity_rate=len(valid)/len(records) if records else None,convergence_rate=converged/len(records) if records else None,
        unique_hypothesis_fits_attempted=nfit,unique_hypothesis_fits_completed=completed,converged_hypothesis_fits=sum(s['converged'] for s in models),
        seconds=elapsed,mean_seconds_per_dataset=elapsed/len(records) if records else None,peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        maximum_GH161_score=max(audit_scores,default=None),maximum_GH101_score=max((s['native_score'] for s in models),default=None),
        maximum_loglik_grid_error=max((s['quadrature_error'] for s in models),default=None),
        maximum_LR_grid_difference=max((abs(row['audit_minus_fit_lr']) for r in records for row in r.get('lr',[])),default=None),
        prior_files_verified=protect(),data_generated=0,p_values_generated=False,old_mc_pooled=False,main_stages_started=False)


def run():
    output=HERE/'results';output.mkdir(exist_ok=True)
    if (output/'metadata.json').exists():raise RuntimeError('Prior pilot evidence protected: no repeated attempts')
    start=time.monotonic();protect();legacy=legacy_records()
    prior_meta=json.loads((LEGACY/'metadata.json').read_text())
    versions={p:importlib.metadata.version(p) for p in ('numpy','scipy','pandas')}
    if any(versions[p]!=prior_meta['versions'][p] for p in versions):raise ValueError('Archived data/engine versions differ')
    pairs=[verify_pair(seed,legacy) for seed in SEEDS]
    E.save(output/'metadata.json',dict(protocol=json.loads((HERE/'protocol.json').read_text()),sources={str(p.relative_to(ROOT)):D.sha(p.read_bytes()) for p in (Path(__file__),HERE/'protocol.json',HERE/'prior_manifest.json',HERE/'audit.py')},
        versions=versions,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),pairing_audit=pairs,prior_files_verified=protect()))
    records=[];current=None;stop='fixed pilot complete';old_handler=signal.getsignal(signal.SIGALRM)
    def alarm(*_):raise PilotStop('Predeclared dataset45s/total600s wall cap')
    signal.signal(signal.SIGALRM,alarm)
    def checkpoint(r):
        nonlocal current
        current=r;E.save(output/'active_case.json',r)
    try:
        for seed in SEEDS:
            for impact in ('minus','zero'):
                remaining=600-(time.monotonic()-start)
                if remaining<=0:raise PilotStop('Total600s budget')
                current=dict(seed=seed,impact=impact,data_sha256=legacy[(seed,impact)]['data_sha256'],valid=False,fits_attempted=0,fits_completed=0,solutions={})
                t=time.monotonic();signal.setitimer(signal.ITIMER_REAL,min(45,remaining))
                case=run_case(seed,impact,legacy[(seed,impact)],checkpoint)
                signal.setitimer(signal.ITIMER_REAL,0);case['wall_seconds']=time.monotonic()-t;case['peak_process_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
                records.append(case);current=None;E.save(output/'progress.json',dict(attempted=len(records),valid=sum(r['valid'] for r in records),seconds=time.monotonic()-start))
                with (output/'records.jsonl').open('a') as stream:stream.write(json.dumps(E.clean(case),allow_nan=False)+'\n');stream.flush()
                (output/'active_case.json').unlink(missing_ok=True)
                print(f'{len(records)}/20 datasets; valid={case["valid"]}; {case["wall_seconds"]:.2f}s; RSS high-water {case["peak_process_rss_mib"]:.1f}MiB',flush=True)
                if not case['valid']:stop=case.get('failure_reason','Technical gate failure');break
                if case['peak_process_rss_mib']>1024:stop='Process RSS safety threshold';break
            if stop!='fixed pilot complete':break
    except Exception as exc:
        stop=f'{type(exc).__name__}: {exc}'
        if current is not None:
            current.update(valid=False,failure_reason=stop,wall_seconds=time.monotonic()-t,peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024);records.append(current)
            with (output/'records.jsonl').open('a') as stream:stream.write(json.dumps(E.clean(current),allow_nan=False)+'\n')
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)
    raw=(output/'records.jsonl').read_bytes() if (output/'records.jsonl').exists() else b'';packed=gzip.compress(raw,mtime=0);assert gzip.decompress(packed)==raw
    (output/'records.jsonl.gz').write_bytes(packed);(output/'records.jsonl').unlink(missing_ok=True)
    summary=summarize(records,time.monotonic()-start,stop);summary.update(records_sha256=D.sha(raw),records_gzip_sha256=D.sha(packed))
    E.save(output/'summary.json',summary)
    from validation.grm_gh101_uniform_pilot.audit import audit
    E.save(output/'audit.json',audit(output))
    print(json.dumps(E.clean(summary),indent=2),flush=True)


if __name__=='__main__':run()
