"""Observed 4PL information from saved estimates only; no optimization calls."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import pandas as pd
from validation.r_mirt_phase1.m12_diagnostic import scores, feasible_scores
from validation.r_mirt_phase1.equivalence_audit import BOUNDS, evaluate


def information(score, p, step):
    coordinates=[(item,k) for item in p.index for k in p.columns]
    base=score(p).to_numpy().ravel()
    matrix=np.empty((len(base),len(base))); stencils=[]
    for j,(item,k) in enumerate(coordinates):
        value=p.loc[item,k];lo,hi=BOUNDS[k]
        h=step*max(1.,abs(value))
        if value==lo or value==hi:
            sign=1 if value==lo else -1
            h=min(h,(hi-lo)/4)
            one,two=p.copy(),p.copy()
            one.loc[item,k]+=sign*h;two.loc[item,k]+=sign*2*h
            derivative=sign*(-3*base+4*score(one).to_numpy().ravel()-score(two).to_numpy().ravel())/(2*h)
            stencil='forward' if sign==1 else 'backward'
        else:
            h=min(h,(value-lo)/2,(hi-value)/2)
            plus,minus=p.copy(),p.copy();plus.loc[item,k]+=h;minus.loc[item,k]-=h
            derivative=(score(plus).to_numpy().ravel()-score(minus).to_numpy().ravel())/(2*h)
            stencil='central'
        matrix[:,j]=-derivative
        stencils.append(dict(item=item,parameter=k,step=float(h),stencil=stencil))
    if not np.isfinite(matrix).all(): raise ValueError('Nonfinite information')
    return matrix,stencils


def stability(first, second):
    a,b=(.5*(x+x.T) for x in (first,second))
    change=float(np.linalg.norm(a-b,2))
    asymmetry=max(float(np.linalg.norm(x-x.T,2)/2) for x in (first,second))
    ea,eb=np.linalg.eigvalsh(a),np.linalg.eigvalsh(b)
    # Derived perturbation criterion, not a scientific acceptance tolerance.
    observed_uncertainty=change+asymmetry
    stable=bool(min(ea[0],eb[0])>observed_uncertainty)
    return dict(coarse_eigenvalues=ea.tolist(),fine_eigenvalues=eb.tolist(),
        step_change_norm2=change,antisymmetric_norm2=asymmetry,
        observed_perturbation_scale=observed_uncertainty,
        positive_definite_both=bool(ea[0]>0 and eb[0]>0),
        positive_definiteness_exceeds_observed_perturbation=stable,
        relative_step_change_norm2=change/float(np.linalg.norm(b,2)),
        condition_number=float(np.linalg.cond(b)))


def modes(matrix, labels):
    eigenvalues,vectors=np.linalg.eigh((matrix+matrix.T)/2)
    result=[]
    for j in range(min(4,len(labels))):
        v=vectors[:,j];order=np.argsort(abs(v))[::-1][:8]
        result.append(dict(eigenvalue=float(eigenvalues[j]),
            M12_squared_loading=float(sum(v[i]**2 for i,l in enumerate(labels) if l.startswith('M12:'))),
            M7_squared_loading=float(sum(v[i]**2 for i,l in enumerate(labels) if l.startswith('M7:'))),
            leading_loadings={labels[i]:float(v[i]) for i in order}))
    return result,eigenvalues,vectors


def audit(source, output):
    source,output=Path(source),Path(output);output.mkdir(parents=True,exist_ok=True)
    report=json.loads((source/'report.json').read_text());record=next(m for m in report['models'] if m['model']=='4PL')
    folder=source/'4PL';ref=folder/record['reference_directory']
    data=pd.read_csv(folder/'data.csv');x=data.to_numpy()
    p=pd.read_csv(ref/'parameters.csv').pivot(index='item',columns='parameter',values='value').loc[data.columns,list(BOUNDS)]
    if not all(p[k].between(lo,hi).all() for k,(lo,hi) in BOUNDS.items()): raise ValueError('Outside physical bounds')
    grid=pd.read_csv(ref/'integration.csv');theta,weights=grid.theta.to_numpy(),grid.weight.to_numpy()
    score=lambda v:scores(x,v,theta,weights)
    g=score(p);step=float(np.cbrt(np.finfo(float).eps))
    coarse,sc=information(score,p,step);fine,sf=information(score,p,step/2)
    coordinates=[(item,k) for item in p.index for k in p.columns]
    active=[i for i,(item,k) in enumerate(coordinates) if p.loc[item,k] in BOUNDS[k]]
    free=[i for i in range(len(coordinates)) if i not in active]
    labels=[f'{item}:{"u" if k=="d" else k}' for item,k in coordinates]
    fc,ff=(h[np.ix_(free,free)] for h in (coarse,fine))
    face=stability(fc,ff);full=stability(coarse,fine)
    mode,eigenvalues,vectors=modes(ff,[labels[i] for i in free])
    diagonal=np.diag(ff)
    scaled={};scaled_vectors=np.empty((0,0))
    if np.all(diagonal>0):
        scaling=1/np.sqrt(diagonal)
        scaled_matrix=scaling[:,None]*ff*scaling[None,:]
        scaled_modes,scaled_eigenvalues,scaled_vectors=modes(scaled_matrix,[labels[i] for i in free])
        scaled=dict(eigenvalues=scaled_eigenvalues.tolist(),condition_number=float(np.linalg.cond(scaled_matrix)),
            modes=scaled_modes,max_diagonally_scaled_face_score=float(max(abs(scaling*g.to_numpy().ravel()[free]))))
    quadratic=dict(status='INCONCLUSIVE',reason='Positive definiteness/stability not established')
    if face['positive_definiteness_exceeds_observed_perturbation'] and face['condition_number']*np.finfo(float).eps<1:
        face_score=g.to_numpy().ravel()[free]
        dc=np.linalg.solve((fc+fc.T)/2,face_score);df=np.linalg.solve((ff+ff.T)/2,face_score)
        candidate=p.copy()
        for i,v in zip(free,df):candidate.loc[coordinates[i]]+=v
        feasible=all(candidate[k].between(lo,hi).all() for k,(lo,hi) in BOUNDS.items())
        quadratic=dict(status='LOCAL_ACTIVE_FACE_PREDICTION_ONLY',feasible=bool(feasible),
            coarse_improvement=float(face_score@dc/2),fine_improvement=float(face_score@df/2),
            displacement={labels[i]:float(v) for i,v in zip(free,df)},
            note='Not a fitted optimum or convergence proof; active bounds held fixed. No candidate likelihood evaluation.')
    ll=evaluate(x,p,theta,weights)[2]
    result=dict(status='LOCAL_INFORMATION_DIAGNOSTIC' if face['positive_definiteness_exceeds_observed_perturbation'] else 'INCONCLUSIVE',new_fits=0,source_run=40151828898,
        original_convergence=record['R'],fit_status='NOT CONVERGED' if not record['R']['converged'] else 'REPORTED_CONVERGED',common_grid_loglik=ll,
        fixed_latent_parameters=['mean=0','variance=1'],coordinates=labels,
        active_bounds={labels[i]:float(p.loc[coordinates[i]]) for i in active},
        full_feasible_one_sided_information=full,active_face_information=face,
        active_face_modes=mode,diagonally_scaled_active_face=scaled,
        gradient=g.to_dict('index'),max_abs_physical_score=float(np.max(abs(g.to_numpy()))),
        max_abs_feasible_score=float(np.max(abs(feasible_scores(p,g).to_numpy()))),
        derivative_stencils=dict(coarse=sc,fine=sf),quadratic_prediction=quadratic,
        caveat='Two step sizes measure sensitivity, not a rigorous error bound. Boundary-normal eigenvectors are not feasible two-sided directions; active-face curvature is conditional, not a full constrained identification proof.',
        source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('information_audit.py','m12_diagnostic.py','equivalence_audit.py')},
        versions={k:importlib.metadata.version(k) for k in ('numpy','scipy','pandas')},
        input_sha256={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (source/'report.json',folder/'data.csv',ref/'parameters.csv',ref/'integration.csv')})
    trace=ref/'convergence_trace_summary.csv'
    if trace.exists():
        result['convergence_trace']=pd.read_csv(trace).iloc[0].to_dict()
        pair=ref/'last_convergence_pair.csv'
        result['convergence_trace']['parameters_not_below_TOL']=int((~pd.read_csv(pair).strictly_below_TOL).sum())
        for file in (trace,pair):result['input_sha256'][str(file.relative_to(source))]=hashlib.sha256(file.read_bytes()).hexdigest()
    np.savez_compressed(output/'information_matrices.npz' ,coarse=coarse,fine=fine,
        face_indices=free,active_indices=active,labels=labels,face_eigenvalues=eigenvalues,
        face_eigenvectors=vectors,scaled_face_eigenvectors=scaled_vectors)
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=audit(a.source,a.output)
    print(json.dumps({k:r[k] for k in ('active_bounds','active_face_information','active_face_modes','diagonally_scaled_active_face','quadratic_prediction')}))
