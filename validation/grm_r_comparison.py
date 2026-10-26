"""Reusable opt-in GRM/PCM calibration pair. One model/attempt; no retries or simulations."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
from scipy.special import log_expit,logsumexp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from validation.poly_r_preflight import ROOT,helpers,HARNESS_COMMIT


def marginal(data,a,b,theta,weights,model="GRM"):
    """Stable GRM respondent MML and physical score; never clips probabilities."""
    if model in ("PCM","GPCM"):
        if model=="PCM" and not np.array_equal(a,np.ones(len(a))):raise ValueError('PCM requires fixed a=1')
        eta=theta[:,None,None]*np.arange(b.shape[1]+1)-np.concatenate([np.zeros((len(a),1)),np.cumsum(b,axis=1)],axis=1)
        logits=a[None,:,None]*eta
        lp=logits-logsumexp(logits,axis=2,keepdims=True)
        onehot=np.eye(b.shape[1]+1)[data]
        joint=np.einsum('njc,qjc->nq',onehot,lp)+np.log(weights)
        norms=logsumexp(joint,axis=1)
        post=np.exp(joint-norms[:,None])
        counts=np.einsum('nq,njc->qjc',post,onehot)
        residual=counts.sum(2,keepdims=True)*np.exp(lp)-counts
        gb=a[:,None]*residual[:,:,::-1].cumsum(2)[:,:,::-1][:,:,1:].sum(0)
        ga=-(residual*eta).sum(axis=(0,2)) if model=='GPCM' else np.zeros(len(a))
        return float(norms.sum()),dict(a=ga,b=gb)
    if model!="GRM":raise ValueError('Only GRM/PCM supported')
    z=a[None,:,None]*(theta[:,None,None]-b[None,:,:])
    ls=log_expit(z)
    middle=ls[:,:,:-1]+np.log(-np.expm1(ls[:,:,1:]-ls[:,:,:-1]))
    lp=np.concatenate([log_expit(-z[:,:,:1]),middle,ls[:,:,-1:]],axis=2)
    onehot=np.eye(b.shape[1]+1)[data]
    joint=np.einsum('njc,qjc->nq',onehot,lp)+np.log(weights)
    norms=logsumexp(joint,axis=1)
    post=np.exp(joint-norms[:,None])
    counts=np.einsum('nq,njc->qjc',post,onehot)
    ld=ls+log_expit(-z)
    pair=counts[:,:,:-1]*np.exp(ld-lp[:,:,:-1])-counts[:,:,1:]*np.exp(ld-lp[:,:,1:])
    gb=a[:,None]*pair.sum(0)
    ga=-np.sum(pair*(theta[:,None,None]-b[None,:,:]),axis=(0,2))
    return float(norms.sum()),dict(a=ga,b=gb)


def python_contract(preflight,model="GRM"):
    import mirt
    from mirt.estimation.base import _ITEM_PARAMETER_BOUNDS
    from mirt.estimation._graded_order import THRESHOLD_GAP
    from mirt.estimation.quadrature import GaussHermiteQuadrature
    if importlib.metadata.version('mirt')!='1.2.0':raise ValueError('Python mirt must be 1.2.0')
    if (_ITEM_PARAMETER_BOUNDS['discrimination'],_ITEM_PARAMETER_BOUNDS['thresholds'],THRESHOLD_GAP)!=((.1,5.),(-6.,6.),1e-6):
        raise ValueError('Python constraints differ')
    grid=pd.read_csv(preflight/'integration.csv',float_precision='round_trip')
    q=GaussHermiteQuadrature(n_points=61,n_dimensions=1)
    if not np.array_equal(q.nodes[:,0],grid.theta) or not np.array_equal(q.weights,grid.weight):
        raise ValueError('Python GH grid does not exactly match prepared nodes/weights')
    if model in ('PCM','GPCM') and _ITEM_PARAMETER_BOUNDS['steps']!=(-6.,6.):raise ValueError('PCM step bounds differ')
    seed=pd.read_csv(preflight/f'{model}_physical.csv',float_precision='round_trip')
    return grid,seed


def standard_normal_population(fit):
    """mirt1.2.0 FitResult: None means zero mean / identity covariance.

    The documented default is normalized, never an estimated scale change.
    Missing attributes or explicitly nonstandard/malformed scales still fail.
    """
    raw_mean,raw_cov=fit.latent_mean,fit.latent_covariance
    mean=np.zeros(1) if raw_mean is None else np.asarray(raw_mean,float)
    cov=np.eye(1) if raw_cov is None else np.asarray(raw_cov,float)
    if not np.array_equal(mean,[0.]) or not np.array_equal(cov,[[1.]]):
        raise ValueError('Python latent distribution differs')
    return dict(raw_latent_mean=None if raw_mean is None else mean.tolist(),
                raw_latent_covariance=None if raw_cov is None else cov.tolist(),
                mean=0,variance=1)


def python_once(data,seed,model="GRM"):
    import mirt
    start={('thresholds' if model=='GRM' else 'steps'):seed[['b1','b2','b3']].to_numpy()}
    if model!='PCM':start['discrimination']=seed.a.to_numpy()
    fit=mirt.fit_mirt(data.to_numpy(int),model=model,n_factors=1,n_categories=4,estimation='EM',
        n_quadpts=61,max_iter=3000,tol=1e-8,priors=None,accelerate='none',
        start_values=start,compute_standard_errors=False)
    population=standard_normal_population(fit)
    if not np.isfinite([fit.log_likelihood,fit.aic,fit.bic]).all():raise ValueError('Nonfinite Python fit diagnostics')
    if getattr(fit,'log_posterior',None) is not None:raise ValueError('Unexpected Python item prior')
    steps=fit.model.thresholds if model=='GRM' else fit.model.steps
    p=pd.DataFrame(dict(item=data.columns,a=fit.model.discrimination,
        **{f'b{i+1}':steps[:,i] for i in range(3)}))
    meta=dict(converged=bool(fit.converged),iterations=int(fit.n_iterations),
        loglik=float(fit.log_likelihood),aic=float(fit.aic),bic=float(fit.bic),
        criterion='absolute marginal logLik increment <1e-8',**population)
    return fit,p,meta


def execute(preflight,output,allow_fit=False,rscript=None,model="GRM"):
    if model not in ("GRM","PCM","GPCM"):raise ValueError("Only GRM/PCM supported")
    preflight,output=Path(preflight).resolve(),Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'fit_started').exists():raise ValueError('Retry refused: use of this fit attempt already recorded')
    result=dict(model=model,runtime_python=sys.version,status='NOT EXECUTED',scientific_pass=False,simulations=0,
        source_sha256={n:hashlib.sha256((ROOT/'validation'/n).read_bytes()).hexdigest() for n in ('grm_r_comparison.py','grm_r_comparison.R')},
        fit_calls=dict(R=0,python=0),execution=dict(R='NOT EXECUTED',python='NOT EXECUTED'))
    def save(): (output/'report.json').write_text(json.dumps(result,indent=2,default=str,allow_nan=False)+'\n')
    save()
    try:
        prior=json.loads((preflight/'report.json').read_text())
        if prior['status']!='PREFLIGHT CHECKED / NOT SCIENTIFICALLY VALIDATED' or prior['R_execution']!='EXECUTED' or not all(x['checked'] for x in prior['R_checks']):
            raise ValueError('Successful actual R preflight required')
        data=pd.read_csv(preflight/'data.csv');h,digest=helpers()
        fixture=ROOT/h.SOURCES['GRM']
        if data.shape!=(500,8) or not data.equals(pd.read_csv(fixture)) or hashlib.sha256(fixture.read_bytes()).hexdigest()!=prior['fixture_sha256']:
            raise ValueError('Dataset differs from verified fixture')
        grid,seed=python_contract(preflight,model)
        if list(seed.item)!=list(data.columns):raise ValueError('Starting item order differs')
        result.update(settings=dict(D=1,ML=True,priors=None,mean=0,variance=1,GH_points=61,
            max_iter=3000,numeric_tol=1e-8,acceleration='none',SE=False,
            bounds=(dict(a=[1,1],b=[-6,6],threshold_ordering=False) if model=="PCM" else
                    dict(a=[.1,5],b=[-6,6],**(dict(minimum_threshold_gap=1e-6) if model=="GRM" else dict(threshold_ordering=False)))),
            python_stop='absolute logLik increment',R_stop='native a/intercept parameter change',
            starting_parameters=seed.to_dict('records')),
            pinned_helpers=dict(commit=HARNESS_COMMIT,sha256=digest),
            versions={k:importlib.metadata.version(k) for k in ('mirt','numpy','scipy','pandas')},
            input_sha256={n:hashlib.sha256((preflight/n).read_bytes()).hexdigest() for n in
                ('report.json','data.csv','integration.csv','evaluation_grid.csv',f'{model}_physical.csv',f'{model}_configured_layout.csv',f'{model}_constraints.csv')})
        executable=rscript or shutil.which('Rscript')
        if not allow_fit or not executable:
            result['reason']='Prepared only; --execute opt-in and an actual Rscript runtime are required';save();return result
        trace=subprocess.run(['git','show',HARNESS_COMMIT+':validation/r_mirt_phase1/optimizer_trace.R'],cwd=ROOT,capture_output=True,text=True,check=True)
        (output/'pr63_optimizer_trace.R').write_text(trace.stdout)
        result['trace_helper_sha256']=hashlib.sha256(trace.stdout.encode()).hexdigest()
        (output/'fit_started').write_text('One attempt only; no retries\n')
        command=[executable,str(ROOT/'validation/grm_r_comparison.R'),str(preflight),str(output),f'EXECUTE_{model}_ONCE',model]
        r=subprocess.run(command,capture_output=True,text=True,timeout=1800)
        (output/'R_stdout.log').write_text(r.stdout);(output/'R_stderr.log').write_text(r.stderr)
        result['execution']['R']='EXECUTED'
        result['fit_calls']['R']=int((output/'R_fit_invoked').exists())
        if r.returncode:raise ValueError(f'R fit process exit {r.returncode}')
        rm=pd.read_csv(output/'R_metadata.csv').iloc[0].to_dict()
        if str(rm['version'])!='1.48' or rm['model']!=model or rm['mean']!=0 or rm['variance']!=1 or not np.isfinite([rm['loglik'],rm['aic'],rm['bic']]).all():raise ValueError('R metadata/specification mismatch; raw CSV preserved')
        result['R']=rm
        with (output/'Python_stdout.log').open('w') as stdout,(output/'Python_stderr.log').open('w') as stderr:
            result['fit_calls']['python']=1;result['execution']['python']='EXECUTED'
            with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):fit,pp,pm=python_once(data,seed,model)
        pp.to_csv(output/'Python_parameters.csv',index=False);result['python']=pm
        rp=pd.read_csv(output/'R_parameters.csv').set_index('item').loc[data.columns].reset_index()
        evaluation=pd.read_csv(preflight/'evaluation_grid.csv');theta=evaluation.theta.to_numpy()
        P=fit.model.probability(theta);info=np.column_stack([fit.model.information(theta,item_idx=i) for i in range(8)])
        py=pd.DataFrame([dict(item=str(item),node=q,theta=t,category=c,probability=P[q,i,c],information=info[q,i])
            for i,item in enumerate(data.columns) for q,t in enumerate(theta) for c in range(4)])
        py.to_csv(output/'Python_curves.csv',index=False)
        result['comparisons']=h.compare_curves(py,pd.read_csv(output/'R_curves.csv'))
        if result['comparisons'].get('status')=='FAIL':raise ValueError('Curve key/schema mismatch')
        # Compare actual probability/info arrays by keys without a scientific cutoff.
        pair=py.merge(pd.read_csv(output/'R_curves.csv'),on=['item','node','category'],suffixes=('_python','_R'),validate='one_to_one')
        if len(pair)!=len(py):raise ValueError('Curve key mismatch')
        result['comparisons']['item_information']=h.discrepancies(pair.information_python,pair.information_R)
        cumulative=pair.sort_values(['item','node','category']).copy()
        for engine in ('python','R'):
            cumulative[engine]=cumulative.groupby(['item','node'])['probability_'+engine].transform(lambda p:p.iloc[::-1].cumsum().iloc[::-1])
        cut=cumulative.category>0
        result['comparisons']['cumulative_ICC']=h.discrepancies(cumulative.loc[cut,'python'],cumulative.loc[cut,'R'])
        for field in ('loglik','aic','bic'):
            result['comparisons']['native_'+field]=h.discrepancies([pm[field]],[rm[field]])
        result['common_grid_loglik']={};result['physical_scores']={};result['constraint_diagnostics']={}
        parameter_frames={}
        for engine,p in [('python',pp),('R',rp)]:
            a=p.a.to_numpy();b=p[['b1','b2','b3']].to_numpy()
            if not np.isfinite(np.c_[a,b]).all() or np.any(a<=0):raise ValueError('Invalid estimates')
            if model=='GRM' and np.any(np.diff(b,axis=1)<=0):raise ValueError('Unordered GRM thresholds')
            if model=='PCM' and not np.array_equal(a,np.ones(len(a))):raise ValueError('PCM discrimination differs from 1')
            ll,score=marginal(data.to_numpy(int),a,b,grid.theta.to_numpy(),grid.weight.to_numpy(),model)
            result['common_grid_loglik'][engine]=ll
            result['physical_scores'][engine]=dict(a=score['a'].tolist(),b=score['b'].tolist(),
                native_a=(score['a']-(score['b']*b/a[:,None]).sum(1)).tolist(),native_d=(-score['b']/a[:,None]).tolist())
            if model=='PCM':
                result['physical_scores'][engine]=dict(b=score['b'].tolist(),native_d=np.c_[-score['b'][:,:-1]+score['b'][:,1:],-score['b'][:,-1]].tolist(),a='FIXED')
            if model=='GPCM':result['physical_scores'][engine]['native_d']=(np.c_[-score['b'][:,:-1]+score['b'][:,1:],-score['b'][:,-1]]/a[:,None]).tolist()
            records=[]
            for i,item in enumerate(data.columns):
                values=([('a fixed',abs(a[i]-1))] if model=='PCM' else [('a lower',.1-a[i]),('a upper',a[i]-5)])
                for j in range(3):values.extend([(f'b{j+1} lower',-6-b[i,j]),(f'b{j+1} upper',b[i,j]-6)])
                if model=='GRM':values.extend([(f'b{j+1} to b{j+2} gap',1e-6-b[i,j+1]+b[i,j]) for j in range(2)])
                records.extend([dict(item=item,constraint=k,residual=float(v)) for k,v in values])
            pd.DataFrame(records).to_csv(output/f'{engine}_physical_constraints.csv',index=False)
            residual=np.array([r['residual'] for r in records])
            result['constraint_diagnostics'][engine]=dict(max_violation=float(max(0.,residual.max())),
                near_active_residuals=np.flatnonzero((abs(residual)<=1e-6)&np.array([r['constraint']!='a fixed' for r in records])).tolist(),
                note='1e-6 proximity label only; exact positive violations remain failures')
            # Match R and Python coordinates: one a, three physical thresholds per item.
            parameters=[dict(a=float(av),b=bv.tolist()) for av,bv in zip(a,b)]
            parameter_frames[engine]=h.parameter_frame(data.columns,parameters)
        result['comparisons']['parameters']=h.compare_parameters(parameter_frames['python'],parameter_frames['R'])
        result['comparisons']['common_grid_loglik']=h.discrepancies([result['common_grid_loglik']['python']],[result['common_grid_loglik']['R']])
        if any(isinstance(v,dict) and v.get('status') in ('FAIL','NOT COMPARABLE') for v in result['comparisons'].values()):raise ValueError('Invalid numerical comparison')
        parameter_pair=parameter_frames['python'].merge(parameter_frames['R'],on=['item','parameter','component'],suffixes=('_python','_R'),validate='one_to_one')
        parameter_pair['estimated']=(parameter_pair.parameter!='a') if model=='PCM' else True
        parameter_pair['absolute_difference']=abs(parameter_pair.value_python-parameter_pair.value_R)
        parameter_pair['relative_difference']=np.divide(parameter_pair.absolute_difference,abs(parameter_pair.value_R),out=np.full(len(parameter_pair),np.nan),where=parameter_pair.value_R!=0)
        parameter_pair.to_csv(output/'parameter_comparison.csv',index=False)
        k=24 if model=='PCM' else 32
        if fit.n_parameters!=k or int(rm['n_parameters'])!=k:raise ValueError('Free parameter counts differ')
        result['fit_statistic_definitions']={}
        for engine,meta in [('python',pm),('R',rm)]:
            ll=result['common_grid_loglik'][engine]
            result['fit_statistic_definitions'][engine]=dict(N=len(data),k=k,
                common_grid_AIC=-2*ll+2*k,common_grid_BIC=-2*ll+np.log(len(data))*k,
                native_AIC_identity_residual=meta['aic']-(-2*meta['loglik']+2*k),
                native_BIC_identity_residual=meta['bic']-(-2*meta['loglik']+np.log(len(data))*k),
                native_minus_common_loglik=meta['loglik']-ll)
        if model=='GPCM' and bool(rm['converged']) and pm['converged']:
            from core.irt_person_abilities import person_abilities
            from core.irt_b14_model_curves import fitted_category_curves
            nodes=grid.theta.to_numpy();weights=grid.weight.to_numpy()
            scoring=fitted_category_curves(dict(result=fit,k=8,items=data.columns,labels=tuple(map(str,range(4)))),nodes)
            ppers=person_abilities(data,scoring,model=model,quadrature=(nodes,weights))
            ppers.to_csv(output/'Python_persons.csv',index=False)
            rpers=pd.read_csv(output/'R_persons.csv')
            if not np.array_equal(rpers.Birey,ppers.Birey) or not np.array_equal(rpers['Ham puan'],ppers['Ham puan']):raise ValueError('Person ordering/raw scores differ')
            persons=ppers.merge(rpers,on='Birey',suffixes=('_python','_R'),validate='one_to_one')
            result['comparisons']['EAP_theta']=h.discrepancies(persons['Yetenek (θ)_python'],persons['Yetenek (θ)_R'])
            result['comparisons']['posterior_SD']=h.discrepancies(persons['SH (θ)_python'],persons['SH (θ)_R'])
            persons.to_csv(output/'person_comparison.csv',index=False)
            rcurves=pd.read_csv(output/'R_GH_curves.csv')
            rcurves['Theta']=nodes[rcurves.node.to_numpy(int)]
            replay=person_abilities(data,rcurves,model=model,quadrature=(nodes,weights))
            result['person_scoring_identity_max_abs']=float(np.abs(replay[['Yetenek (θ)','SH (θ)']].to_numpy()-rpers[['Yetenek (θ)','SH (θ)']].to_numpy()).max())
            if not np.allclose(replay[['Yetenek (θ)','SH (θ)']],rpers[['Yetenek (θ)','SH (θ)']],rtol=np.sqrt(np.finfo(float).eps),atol=np.sqrt(np.finfo(float).eps)):
                raise ValueError('Independent posterior integration identity failed')
            result['person_settings']=dict(method='EAP',prior='N(0,1)',nodes=61,reference='Independent R posterior from fitted mirt probtrace; not default fscores quadrature')
        result['R_minus_python_common_grid_loglik']=result['common_grid_loglik']['R']-result['common_grid_loglik']['python']
        result['status']='MEASURED / NOT CERTIFIED'
        if any(x['max_violation']>0 for x in result['constraint_diagnostics'].values()):raise ValueError('Positive physical constraint violation')
        if not bool(rm['converged']) or not pm['converged']:result.update(status='NOT CONVERGED',reason='At least one backend did not converge')
        result['limitations']=['Different optimizer/stopping criteria; convergence flags are not proof of stationarity/global optimality.','No calibrated scientific discrepancy cutoff or SE/CI equivalence.']
    except subprocess.TimeoutExpired as error:
        result.update(status='FAIL',reason='R timeout; no retry')
        result['execution']['R']='EXECUTED';result['fit_calls']['R']=int((output/'R_fit_invoked').exists())
        for name,value in [('R_stdout.log',error.stdout),('R_stderr.log',error.stderr)]:
            (output/name).write_text(value.decode(errors='replace') if isinstance(value,bytes) else value or '')
    except Exception as error:
        result.update(status='FAIL',reason=f'{type(error).__name__}: {error}')
    save();return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--model',choices=['GRM','PCM','GPCM'],default='GRM')
    a=p.parse_args();r=execute(a.preflight,a.output,a.execute,model=a.model);print(r['status'])
    if r['status'] in ('FAIL','NOT CONVERGED'):raise SystemExit(1)
