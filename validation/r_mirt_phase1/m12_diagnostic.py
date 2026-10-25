"""No refits: physical marginal scores and feasible M12 likelihood directions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import log_expit, logsumexp
from validation.r_mirt_phase1.equivalence_audit import evaluate, BOUNDS
from validation.r_mirt_phase1.harness import discrepancies


def scores(x, p, theta, weights):
    a,b,c,u=(p[k].to_numpy() for k in ('a','b','c','d'))
    z=(theta[:,None]-b)*a
    ls,lt=log_expit(z),log_expit(-z)
    with np.errstate(divide='ignore'):
        lw=np.log(u-c)
        lp=np.logaddexp(np.log(c),lw+ls)
        lq=np.logaddexp(np.log1p(-u),lw+lt)
    joint=x@lp.T+(1-x)@lq.T+np.log(weights)
    post=np.exp(joint-logsumexp(joint,axis=1,keepdims=True))
    n1,n0=post.T@x,post.T@(1-x)
    mid=n1*np.exp(lw+ls+lt-lp)-n0*np.exp(lw+ls+lt-lq)
    return pd.DataFrame(dict(a=np.sum(mid*(theta[:,None]-b),axis=0),
        b=np.sum(-a*mid,axis=0),
        c=np.sum(n1*np.exp(lt-lp)-n0*np.exp(lt-lq),axis=0),
        d=np.sum(n1*np.exp(ls-lp)-n0*np.exp(ls-lq),axis=0)),index=p.index)


def feasible_scores(parameters, gradient):
    """Box-tangent ascent score: discard only directions blocked at exact bounds."""
    result=gradient.copy()
    for k,(lo,hi) in BOUNDS.items():
        result[k]=np.where(parameters[k]==lo,np.maximum(gradient[k],0),
            np.where(parameters[k]==hi,np.minimum(gradient[k],0),gradient[k]))
    return result


def diagnose(source):
    source=Path(source)
    report=json.loads((source/'report.json').read_text())
    record=next(m for m in report['models'] if m['model']=='4PL')
    folder=source/'4PL'; reference=folder/record['reference_directory']
    data=pd.read_csv(folder/'data.csv'); x=data.to_numpy()
    grid=pd.read_csv(reference/'integration.csv'); theta,weights=grid.theta.to_numpy(),grid.weight.to_numpy()
    frames={'python':pd.read_csv(folder/'python_parameters.csv'), 'R':pd.read_csv(reference/'parameters.csv')}
    result=dict(status='NOT CONVERGED' if not record['R']['converged'] else 'DIAGNOSTIC_NOT_CERTIFIED',refits=0,
        source_report_sha256=hashlib.sha256((source/'report.json').read_bytes()).hexdigest(),
        models={'4PL':{}},note='Directional steps are likelihood evaluations, not optimizations or acceptance thresholds.')
    curves={}
    fitted={}
    for engine,f in frames.items():
        p=f.pivot(index='item',columns='parameter',values='value').loc[data.columns]
        fitted[engine]=p
        gradient=scores(x,p,theta,weights)
        projected=feasible_scores(p,gradient)
        ll=evaluate(x,p,theta,weights)[2]
        m=p.loc['M12']; g=gradient.loc['M12']
        conditional=[]
        for name in p.columns:
            h=np.cbrt(np.finfo(float).eps)*max(1.,abs(m[name]))
            lo,hi=BOUNDS[name]; h=min(h,(m[name]-lo)/2,(hi-m[name])/2)
            if h<=0:
                conditional=[];break  # No central physical stencil at an exact bound.
            plus,minus=p.copy(),p.copy()
            plus.loc['M12',name]+=h;minus.loc['M12',name]-=h
            conditional.append(-(scores(x,plus,theta,weights).loc['M12']-scores(x,minus,theta,weights).loc['M12']).to_numpy()/(2*h))
        information=np.column_stack(conditional) if conditional else None
        eigenvalues=np.linalg.eigvalsh((information+information.T)/2) if information is not None else np.array([])

        # Native R coordinates (a,intercept,g_logit,u_logit), b=-intercept/a.
        native=dict(a=float(g.a-g.b*m.b/m.a),intercept=float(-g.b/m.a),
            g_logit=float(g.c*m.c*(1-m.c)),u_logit=float(g.d*m.d*(1-m.d)))
        steps=[]
        for delta in (-.000001,-.0001,-.001,-.01):
            q=p.copy();q.loc['M12','d']+=delta
            if q.loc['M12','d']>=.5:
                value=evaluate(x,q,theta,weights)[2]
                steps.append(dict(delta_u=delta,loglik_gain=value-ll,first_order_prediction=float(g.d*delta)))
        # Fixed diagnostic probes, not optimizer steps or acceptance tolerances.
        direction=projected/max(float(np.abs(projected.to_numpy()).max()),1.)
        ascent=[]
        for step in (1e-6,1e-4,1e-3):
            q=p+step*direction
            if all(q[k].between(lo,hi).all() for k,(lo,hi) in BOUNDS.items()):
                ascent.append(dict(step=step,loglik_gain=evaluate(x,q,theta,weights)[2]-ll,
                    first_order_prediction=float(step*(gradient*direction).to_numpy().sum())))
        display=np.linspace(-4,4,81)
        prob,info,_=evaluate(x,p,display,np.full(81,1/81))
        curves[engine]=(prob[:,list(p.index).index('M12')],info)
        result['models']['4PL'][engine]=dict(original_converged=record[engine]['converged'],
            original_iterations=record[engine].get('iterations'), common_grid_loglik=ll,
            M12_conditional_physical_information_eigenvalues=eigenvalues.tolist(),
            M12_parameters=m.to_dict(),M12_physical_score=g.to_dict(),M12_native_R_score=native,
            all_physical_scores=gradient.to_dict('index'),feasible_upper_steps=steps,
            box_tangent_scores=projected.to_dict('index'),
            max_abs_box_tangent_score=float(np.abs(projected.to_numpy()).max()),
            feasible_ascent_probes=ascent,
            M12_constraints={k:dict(value=float(m[k]),lower=lo,upper=hi) for k,(lo,hi) in BOUNDS.items()})
    base=fitted['R']
    ll=evaluate(x,base,theta,weights)[2]
    result['R_fixed_other_items_M12_path_to_python']=[]
    for fraction in (.001,.01,.1,1.):
        p=base.copy()
        p.loc['M12']=base.loc['M12']+fraction*(fitted['python'].loc['M12']-base.loc['M12'])
        result['R_fixed_other_items_M12_path_to_python'].append(dict(fraction=fraction,
            loglik_gain=evaluate(x,p,theta,weights)[2]-ll))
    result['input_sha256']={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest()
        for f in (folder/'data.csv',folder/'python_parameters.csv',reference/'parameters.csv',reference/'integration.csv')}
    result['original_boundary_flags']=record['boundary_flags']
    result['R_original_diagnostics']=record['R']
    result['R_stdout']=(folder/'R_stdout.txt').read_text()
    result['R_stderr']=(folder/'R_stderr.txt').read_text()
    result['limitation']='Original artifact has no per-iteration history or per-M-step NLopt status; exact inner-optimizer stopping cause cannot be inferred.'
    result['R_minus_python_common_grid_loglik']=result['models']['4PL']['R']['common_grid_loglik']-result['models']['4PL']['python']['common_grid_loglik']
    result['M12_ICC']=discrepancies(curves['python'][0],curves['R'][0])
    result['test_information']=discrepancies(curves['python'][1],curves['R'][1])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.write_text(json.dumps(diagnose(a.source),indent=2,allow_nan=False))
