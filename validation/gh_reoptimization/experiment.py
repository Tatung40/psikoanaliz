"""One saved dataset, versioned GH101-first feasibility; no MC/p production."""
import importlib.metadata
import json
from pathlib import Path
import resource
import signal
import subprocess
import time

import numpy as np
import pandas as pd
from core import grm_dif as G
from validation.gh81_diagnostic import diagnostic as D

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
    if isinstance(v,(bool,np.bool_)):return bool(v)
    if isinstance(v,(int,np.integer)):return int(v)
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    return v


def save(p,v):
    p.write_text(json.dumps(clean(v),indent=2,allow_nan=False)+'\n')


def protect():
    m=json.loads((HERE/'prior_manifest.json').read_text())
    for p,h in m.items():
        if D.sha((ROOT/p).read_bytes())!=h:raise ValueError('Prior source/evidence changed: '+p)
    return len(m)


def saved_models(record):
    models={};slots={}
    for item,fits in record['result']['fits'].items():
        for h,f in fits.items():
            design,p,error=D.recover(item,h,f)
            key=D.sha(json.dumps(design.roles).encode()+p.tobytes());slots[f'{item}.{h}']=key
            if key not in models:models[key]=dict(design=design,params=p,loglik=f['loglik'],slot=f'{item}.{h}')
    return models,slots


def anchor_feasible(f):
    a,b,mu,sd=f['design'].unpack(f['params'])
    return bool(np.array_equal(a[0,:8],a[1,:8]) and np.array_equal(b[0,:8],b[1,:8]) and np.all(np.diff(b,axis=2)>0))


def dense_information(p,design,parts,grid,n):
    matrix=np.empty((design.size,design.size))
    for j in range(design.size):
        h=1e-5*max(1.,abs(p[j]));step=np.zeros(design.size);step[j]=h
        matrix[:,j]=n*(G.objective(p+step,design,parts,grid)[1]-G.objective(p-step,design,parts,grid)[1])/(2*h)
    asymmetry=float(np.max(np.abs(matrix-matrix.T)));matrix=(matrix+matrix.T)/2
    eigen=np.linalg.eigvalsh(matrix)
    good=bool(np.isfinite(eigen).all() and eigen[0]>1e-7*max(eigen[-1],1.))
    return dict(valid=good,eigenvalues=eigen,min_eigenvalue=eigen[0],condition=eigen[-1]/eigen[0] if eigen[0]>0 else None,max_asymmetry_before_symmetrizing=asymmetry)


def lr_table(slots,results,fit_q,audit_q):
    rows=[]
    for item in ('P09','P10'):
        for label,null,df in (('omnibus','H0',4),('thresholds','Ha',3),('slope','Hb',1)):
            alt=results[slots[item+'.H2']];base=results[slots[item+'.'+null]]
            stats={};notes={}
            for q in sorted({fit_q,audit_q,161}):
                stat,note=G._nested_lr(alt['evaluations'][str(q)]['loglik'],base['evaluations'][str(q)]['loglik'])
                stats[str(q)]=stat;notes[str(q)]=note
            rows.append(dict(item=item,test=label,df=df,lr=stats,native_nested_notes=notes,
                audit_minus_fit_lr=stats[str(audit_q)]-stats[str(fit_q)],
                implied_grid_error_bound=4e-3,valid=not any(notes.values())))
    return rows


class BudgetStop(Exception):pass


def run():
    protect();protocol=json.loads((HERE/'protocol.json').read_text());record,x,g,source=D.load_saved();models,slots=saved_models(record)
    columns=[f'P{i:02d}' for i in range(1,11)]
    inputs=G.validate_inputs(pd.DataFrame(x,columns=columns),g,0,1,columns[:8],categories=4,min_cell=5)
    output=HERE/'results';output.mkdir(exist_ok=True)
    if (output/'report.json').exists():raise RuntimeError('Existing evidence protected: do not repeat fits')
    baseline=json.loads((D.HERE/'results/diagnostic.json').read_text())
    start=time.monotonic();report=dict(protocol=protocol,slots=slots,stages=[],source_archive=source.name,
        data_sha256=record['data_sha256'],source_archive_sha256=D.sha(source.read_bytes()),fits_attempted=0,fits_executed=0,p_values_generated=False,
        versions={p:importlib.metadata.version(p) for p in ('numpy','scipy','pandas')},
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={str(p.relative_to(ROOT)):D.sha(p.read_bytes()) for p in (Path(__file__),HERE/'protocol.json',HERE/'prior_manifest.json')})
    def alarm(*_):raise BudgetStop('Predeclared wall limit reached')
    previous_handler=signal.signal(signal.SIGALRM,alarm)
    def arm(seconds=120):
        remaining=120-(time.monotonic()-start)
        if remaining<=0:raise BudgetStop('Total120s budget exhausted')
        signal.setitimer(signal.ITIMER_REAL,min(seconds,remaining))
    previous={};status='NO COMPLETED STAGE'
    try:
        arm()
        for specification in protocol['stages']:
            fit_q,audit_q=specification['fit_points'],specification['audit_points']
            fitter=G.Fitter(inputs,points=fit_q)
            fitter.audit_grid=D.reference_grid(audit_q)  # Instance configuration ONLY; native code unchanged.
            stage=dict(fit_points=fit_q,audit_points=audit_q,solutions={},lr=[],passed=False)
            report['stages'].append(stage);native={}
            for key,old in models.items():
                design=old['design'];initial=old
                if fit_q==161 and key in previous:
                    candidate=previous[key];lo,hi=np.array(design.bounds).T
                    if np.isfinite(candidate['params']).all() and np.min(np.minimum(candidate['params']-lo,hi-candidate['params']))>=1e-4:initial=candidate
                lo,hi=np.array(design.bounds).T
                if not np.array_equal(np.clip(initial['params'],lo,hi),initial['params']):raise ValueError('Would clip start; do not silently alter initialization')
                report['fits_attempted']+=1
                report['active_attempt']=dict(solution=key,slot=old['slot'],fit_points=fit_q,initial_params=initial['params'].tolist())
                save(output/'progress.json',report)
                t=time.monotonic();arm(30)
                fitted=fitter.fit(design.roles,start_fit=initial,information=True,max_iter=2000)
                arm();duration=time.monotonic()-t;report['fits_executed']+=1;native[key]=fitted
                p=fitted['params'];evaluations={}
                for q in sorted({fit_q,audit_q,161}):
                    grid=G._grid(q) if q==fit_q else D.reference_grid(q)
                    value,gradient=G.objective(p,design,fitter.parts,grid);j=int(np.argmax(np.abs(gradient)))
                    evaluations[str(q)]=dict(loglik=-len(x)*value,max_abs_normalized_gradient=abs(gradient[j]),max_parameter=D.labels(design)[j],gradient=gradient,
                        independent_loglik=-len(x)*D.reference_nll(p,design,x,g,q),
                        weight_sum=float(np.exp(grid[1]).sum()),second_moment=float(np.exp(grid[1])@(grid[0]*grid[0])))
                information=dense_information(p,design,fitter.parts,D.reference_grid(audit_q),len(x))
                derivative_checks={};derivatives_ok=True
                try:
                    for h in (1e-4,1e-5):
                        numeric,_=D.finite_difference(p,design,x,g,audit_q,h);analytic=np.asarray(evaluations[str(audit_q)]['gradient'])
                        derivative_checks[str(h)]=dict(max_abs_error=float(np.max(np.abs(numeric-analytic))),finite_difference_gradient=numeric)
                        derivatives_ok &= bool(np.allclose(numeric,analytic,atol=1e-8,rtol=1e-5))
                except ValueError as exc:
                    derivative_checks['reason']=str(exc);derivatives_ok=False
                a,b,mu,sd=design.unpack(p)
                native_ok=bool(fitted['converged'] and not fitted['at_bound'] and fitted['quadrature_ok'] and fitted['cov'] is not None)
                result=dict(first_slot=old['slot'],roles=design.roles,initial_params=initial['params'],params=p,
                    initial_source='GH101' if initial is not old else 'original GH61',a=a,b=b,focal_mean=mu,focal_sd=sd,
                    seconds=duration,iterations=fitted['iterations'],converged=fitted['converged'],at_bound=fitted['at_bound'],
                    native_score=fitted['score'],native_audit_score=fitted['audit_score'],quadrature_error=fitted['quadrature_error'],quadrature_ok=fitted['quadrature_ok'],
                    native_information_min_eigenvalue=fitted['information_min_eigenvalue'],native_information_condition=fitted['information_condition'],covariance_available=fitted['cov'] is not None,
                    reason=fitted['reason'],evaluations=evaluations,dense_information=information,derivative_checks=derivative_checks,
                    anchor_equality=anchor_feasible(fitted),derivatives_ok=derivatives_ok,
                    passed=bool(native_ok and information['valid'] and derivatives_ok and anchor_feasible(fitted)))
                stage['solutions'][key]=result;report.pop('active_attempt',None)
                report['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
                save(output/'progress.json',report)
                if report['peak_rss_mib']>1024:raise BudgetStop('Predeclared1024MiB safety threshold')
            stage['lr']=lr_table(slots,stage['solutions'],fit_q,audit_q)
            stage['passed']=all(s['passed'] for s in stage['solutions'].values()) and all(r['valid'] for r in stage['lr'])
            stage['native_optimizer_calls']=fitter.optimizations
            # Separate integration and optimization effects on the same GH161 grid.
            for row in stage['lr']:
                null={'omnibus':'H0','thresholds':'Ha','slope':'Hb'}[row['test']];item=row['item']
                def old_ll(h,q):
                    ref=baseline['slots'][item+'.'+h];return baseline['unique_solutions'][ref['solution_id']]['evaluations'][str(q)]['loglik']
                old61,_=G._nested_lr(old_ll('H2',61),old_ll(null,61));old161,_=G._nested_lr(old_ll('H2',161),old_ll(null,161))
                row.update(original_GH61_lr=old61,original_fixed_GH161_lr=old161,new_common_GH161_minus_original_GH161_lr=row['lr']['161']-old161)
            save(output/'progress.json',report)
            if stage['passed']:status=f'GH{fit_q} TECHNICAL FEASIBILITY PASSED';break
            previous=native;status=f'GH{fit_q} stage did not pass'
    except Exception as exc:
        status=f'INCOMPLETE / {type(exc).__name__}: {exc}'
        report['error']=str(exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous_handler)
    report.update(status=status,seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        prior_files_verified=protect(),scientific_classification='SINGLE DATASET / NOT CERTIFIED',previous_mc_unchanged=True)
    save(output/'report.json',report);(output/'progress.json').unlink(missing_ok=True)
    print(json.dumps(clean({k:v for k,v in report.items() if k not in ('protocol','sources','stages','slots')}),indent=2),flush=True)


if __name__=='__main__':run()
