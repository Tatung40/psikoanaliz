"""Fixed-solution integration/gradient diagnostic. Optimizer calls forbidden."""
from functools import lru_cache
import gzip
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import numpy as np
from scipy.special import expit, log_expit, logsumexp, roots_hermitenorm
from core import grm_dif as G

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PRIOR=ROOT/'validation/grm_lr_full_mc/results'
POINTS=(61,81,101,121,161)
STEPS=(1e-4,1e-5)


def sha(raw):
    import hashlib
    return hashlib.sha256(raw).hexdigest()


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def protect():
    m=json.loads((HERE/'prior_manifest.json').read_text())
    for p,h in m.items():
        if sha((ROOT/p).read_bytes())!=h:raise ValueError('Protected source/evidence changed: '+p)
    return len(m)


def load_saved():
    for path in reversed(sorted(PRIOR.glob('chunk_*.jsonl.gz'))):
        for line in gzip.decompress(path.read_bytes()).splitlines():
            record=json.loads(line)
            if record['seed']==2026110981 and record['impact']=='minus':
                with np.load(PRIOR/record['failed_matrix']) as data:x,g=data['X'].copy(),data['group'].copy()
                assert sha((PRIOR/record['failed_matrix']).read_bytes())==record['matrix_sha256']
                assert sha(x.tobytes()+g.tobytes())==record['data_sha256']
                return record,x,g,path
    raise ValueError('Saved failed solution missing; no regeneration/refit permitted')


def recover(item,hypothesis,fit):
    roles=['anchor']*8+['ab']*2
    if hypothesis!='H2':roles[int(item[1:])-1]={'H0':'anchor','Hb':'b','Ha':'a'}[hypothesis]
    d=G.Design(roles,4)
    p=d.encode(np.array(fit['a']),np.array(fit['b']),fit['focal_mean'],fit['focal_sd'])
    a,b,mu,sd=d.unpack(p)
    error=max(float(np.max(np.abs(a-fit['a']))),float(np.max(np.abs(b-fit['b']))),abs(mu-fit['focal_mean']),abs(sd-fit['focal_sd']))
    return d,p,error


def labels(d,physical=False):
    result=[]
    for kind,g,i,c in d.labels:
        group={-1:'shared',0:'reference',1:'focal'}[g]
        coordinate='a' if kind=='a' else f'b{c+1}' if physical or c==0 else f'log_gap_b{c+1}'
        result.append(f'P{i+1:02d}.{group}.{coordinate}')
    return result+['focal.mean','focal.sd' if physical else 'focal.log_sd']


def physical_score(score,p,d):
    result=np.array(score,copy=True);blocks={}
    for j,(kind,g,i,c) in enumerate(d.labels):
        if kind=='b':blocks.setdefault((g,i),[]).append(j)
    for indices in blocks.values():
        result[indices]=0.;result[indices[0]]=score[indices[0]]
        for j,previous in zip(indices[1:],indices[:-1]):
            term=score[j]/np.exp(p[j]);result[j]+=term;result[previous]-=term
    result[-1]/=np.exp(p[-1])
    return result


@lru_cache(None)
def reference_grid(q):
    z,w=roots_hermitenorm(q)
    return z,np.log(w)-.5*np.log(2*np.pi)


def reference_nll(p,d,x,groups,q):
    """Direct respondent likelihood, no native category/gradient/pattern code."""
    a,b,mu,sd=d.unpack(p);z,lw=reference_grid(q);ll=0.
    for group in (0,1):
        rows=x[groups==group];theta=z if group==0 else mu+sd*z
        joint=np.broadcast_to(lw,(len(rows),q)).copy()
        for i in range(x.shape[1]):
            eta=a[group,i]*(theta[:,None]-b[group,i])
            # Stable GRM CDF differences; no probability clipping/floor.
            middle=log_expit(eta[:,:-1])+log_expit(-eta[:,1:])+np.log(-np.expm1(eta[:,1:]-eta[:,:-1]))
            lp=np.column_stack((log_expit(-eta[:,0]),middle,log_expit(eta[:,-1])))
            observed=rows[:,i]>=0
            joint[observed]+=lp[:,rows[observed,i]].T
        ll+=float(logsumexp(joint,axis=1).sum())
    return -ll/len(x)


def finite_difference(p,d,x,g,q,step):
    value=reference_nll(p,d,x,g,q);score=[];sensitivity=[];bounds=np.array(d.bounds)
    for j in range(len(p)):
        h=step*max(1.,abs(p[j]));plus=p.copy();minus=p.copy();plus[j]+=h;minus[j]-=h
        if plus[j]>bounds[j,1] or minus[j]<bounds[j,0]:raise ValueError('Central difference crosses a bound; do not silently approximate')
        fp=reference_nll(plus,d,x,g,q);fm=reference_nll(minus,d,x,g,q)
        score.append((fp-fm)/(2*h));sensitivity.append(dict(h=h,loglik_plus_change=-len(x)*(fp-value),loglik_minus_change=-len(x)*(fm-value)))
    return np.array(score),sensitivity


def run():
    start=time.monotonic();protected=protect();record,x,g,path=load_saved();parts=G.pattern_parts(x,g,4);out=HERE/'results';out.mkdir(exist_ok=True)
    if (out/'diagnostic.json').exists():raise RuntimeError('Existing diagnostic evidence protected; do not overwrite')
    unique={};slots={}
    # These guards are effective even if a future edit accidentally calls a fit API.
    with patch.object(G,'minimize',side_effect=RuntimeError('REFIT FORBIDDEN')),patch.object(G.Fitter,'fit',side_effect=RuntimeError('REFIT FORBIDDEN')):
        for item,fits in record['result']['fits'].items():
            for hypothesis,fit in fits.items():
                d,p,error=recover(item,hypothesis,fit)
                key=sha(json.dumps(d.roles).encode()+p.tobytes());slot=f'{item}.{hypothesis}'
                slots[slot]=dict(solution_id=key,roundtrip_physical_max_error=error,saved_loglik=fit['loglik'],saved_score=fit['score'],saved_audit_score=fit['audit_score'])
                if key in unique:continue
                name=labels(d);physical_names=labels(d,True);evaluations={}
                for q in POINTS:
                    value,score=G.objective(p,d,parts,G._grid(q));physical=physical_score(score,p,d)
                    j=int(np.argmax(np.abs(score)));jp=int(np.argmax(np.abs(physical)))
                    z,lw=G._grid(q);rz,rlw=reference_grid(q)
                    independent=reference_nll(p,d,x,g,q)
                    evaluations[str(q)]=dict(loglik=float(-len(x)*value),normalized_gradient=score.tolist(),
                        max_abs_gradient=float(abs(score[j])),max_parameter=name[j],signed_max_gradient=float(score[j]),total_loglik_score_at_max=float(-len(x)*score[j]),
                        physical_gradient=physical.tolist(),physical_max_abs_gradient=float(abs(physical[jp])),physical_max_parameter=physical_names[jp],
                        native_gate_exceeded=bool(np.max(np.abs(score))>1e-5),
                        independent_loglik=float(-len(x)*independent),native_minus_independent_loglik=float(len(x)*(independent-value)),
                        quadrature_moments=dict(weight_sum=float(np.exp(lw).sum()),mean=float(np.exp(lw)@z),second_moment=float(np.exp(lw)@(z*z)),native_vs_reference_node_max_diff=float(np.max(np.abs(z-rz))),native_vs_reference_weight_max_diff=float(np.max(np.abs(np.exp(lw)-np.exp(rlw))))))
                for q,v in evaluations.items():
                    v['loglik_minus_GH61']=v['loglik']-evaluations['61']['loglik']
                    v['loglik_minus_GH161']=v['loglik']-evaluations['161']['loglik']
                    difference=np.array(v['normalized_gradient'])-evaluations['161']['normalized_gradient']
                    v['gradient_minus_GH161_max_abs']=float(np.max(np.abs(difference)))
                    v['gradient_change_max_parameter']=name[int(np.argmax(np.abs(difference)))]
                checks={}
                for q in (61,161):
                    analytic=np.array(evaluations[str(q)]['normalized_gradient']);checks[str(q)]={}
                    for h in STEPS:
                        numeric,sensitivity=finite_difference(p,d,x,g,q,h);err=np.abs(analytic-numeric);j=int(np.argmax(err))
                        checks[str(q)][str(h)]=dict(gradient=numeric.tolist(),max_abs_error=float(err[j]),worst_parameter=name[j],per_parameter_sensitivity=sensitivity)
                unique[key]=dict(first_slot=slot,roles=list(d.roles),n_parameters=d.size,parameters=p.tolist(),labels=name,physical_labels=physical_names,
                    evaluations=evaluations,finite_differences=checks,saved_loglik_reconstruction_error=evaluations['61']['loglik']-fit['loglik'],saved_score_reconstruction_error=evaluations['61']['max_abs_gradient']-fit['score'],saved_audit_score_reconstruction_error=evaluations['81']['max_abs_gradient']-fit['audit_score'])
    result=dict(seed=record['seed'],impact=-.5,n=len(x),shape=list(x.shape),data_sha256=record['data_sha256'],source_archive=path.name,
        source_archive_sha256=sha(path.read_bytes()),source_failed_matrix=record['failed_matrix'],slots=slots,unique_solutions=unique,
        settings=json.loads((HERE/'plan.json').read_text()),seconds=time.monotonic()-start,prior_files_verified=protect(),fits_executed=0,
        versions={p:importlib.metadata.version(p) for p in ('numpy','scipy','pandas')},
        sources={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in (Path(__file__),HERE/'plan.json',HERE/'prior_manifest.json')},
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    save(out/'diagnostic.json',result)
    print(json.dumps(dict(unique_solutions=len(unique),slots=len(slots),seconds=result['seconds'],fits_executed=0,protected=protected)),flush=True)
    for slot,ref in slots.items():
        u=unique[ref['solution_id']]
        print(json.dumps(dict(slot=slot,grids={q:dict(score=v['max_abs_gradient'],parameter=v['max_parameter'],delta_loglik=v['loglik_minus_GH161']) for q,v in u['evaluations'].items()})),flush=True)


if __name__=='__main__':run()
