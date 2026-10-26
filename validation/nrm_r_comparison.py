"""NRM-specific adapter for PR72's independent reference workflow; no estimator."""
import contextlib,hashlib,importlib.metadata,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from validation.poly_r_preflight import ROOT,helpers,HARNESS_COMMIT
from validation.grm_r_comparison import python_contract,standard_normal_population,run_r_process
from core.irt_person_abilities import person_abilities, library_person_abilities
from core.irt_b14_model_curves import fitted_category_curves


def orientation(A,start):
    contrast=float(np.sum(A*start));scale=float(np.linalg.norm(A)*np.linalg.norm(start))
    if not scale or abs(contrast)<=np.sqrt(np.finfo(float).eps)*scale:raise ValueError('NRM orientation contrast unresolved')
    return 1 if contrast>0 else -1


def marginal(data,A,D,nodes,weights):
    logits=nodes[:,None,None]*A[None]+D[None]
    lp=logits-logsumexp(logits,axis=2,keepdims=True)
    one=np.eye(A.shape[1])[data]
    joint=np.einsum('nic,qic->nq',one,lp)+np.log(weights)
    ll=logsumexp(joint,axis=1);post=np.exp(joint-ll[:,None])
    counts=np.einsum('nq,nic->qic',post,one)
    residual=counts-counts.sum(2,keepdims=True)*np.exp(lp)
    return float(ll.sum()),dict(A=np.einsum('qic,q->ic',residual,nodes),D=residual.sum(0))


def parameters(p):
    return p[[f'A{i}' for i in range(4)]].to_numpy(),p[[f'D{i}' for i in range(4)]].to_numpy()


def frame(items,A,D):
    return pd.DataFrame(dict(item=items,**{f'A{i}':A[:,i] for i in range(4)},**{f'D{i}':D[:,i] for i in range(4)}))


def acceptance(metrics,sem_relative,n_items,criteria):
    limits=dict(criteria['max_absolute'])
    limits.update(test_information=n_items*limits['item_information'],aic=2*limits['loglik'],bic=2*limits['loglik'])
    checks={k:dict(value=float(metrics[k]),limit=float(v),passed=bool(np.isfinite(metrics[k]) and metrics[k]<=v)) for k,v in limits.items()}
    checks['SEM_relative']=dict(value=float(sem_relative),limit=criteria['SEM_max_relative'],passed=bool(np.isfinite(sem_relative) and sem_relative<=criteria['SEM_max_relative']))
    return checks


def execute(preflight,output,allow_fit=False,rscript=None):
    pre,out=Path(preflight).resolve(),Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'fit_started').exists():raise ValueError('Retry forbidden in the same attempt')
    criteria_path=ROOT/'validation/nrm_acceptance.json';criteria=json.loads(criteria_path.read_text())
    r=dict(model='NRM',status='NOT EXECUTED',decision='BLOCKED',scientific_pass=False,
        fit_calls=dict(R=0,python=0),execution=dict(R='NOT EXECUTED',python='NOT EXECUTED'),
        criteria=criteria,criteria_sha256=hashlib.sha256(criteria_path.read_bytes()).hexdigest(),
        runtime_python=sys.version,versions={k:importlib.metadata.version(k) for k in ('mirt','numpy','scipy','pandas')})
    def save(): (out/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False,default=str)+'\n')
    save()
    try:
        previous=json.loads((pre/'report.json').read_text())
        if previous['R_execution']!='EXECUTED' or previous['status']!='PREFLIGHT CHECKED / NOT SCIENTIFICALLY VALIDATED':raise ValueError('Actual successful shared R preflight required')
        h,_=helpers();x=pd.read_csv(pre/'data.csv');fixture=ROOT/h.SOURCES['NRM']
        original=pd.read_csv(fixture)
        labels=['A','B','C','D'];mapped=original.replace(dict(zip(labels,range(4)))).astype(int)
        if not x.equals(mapped) or x.shape!=(500,8):raise ValueError('NRM data/category mapping differs from original fixture')
        original.to_csv(out/'nominal_data.csv',index=False);x.to_csv(out/'data.csv',index=False)
        pd.DataFrame(dict(label=labels,code=range(4))).to_csv(out/'category_mapping.csv',index=False)
        grid,_=python_contract(pre);nodes=grid.theta.to_numpy();weights=grid.weight.to_numpy()
        model=__import__('mirt').NominalResponseModel(8,4)
        from mirt.estimation.base import _parameter_bounds
        if _parameter_bounds(model,'slopes')!=(-5.,5.) or _parameter_bounds(model,'intercepts')!=(-6.,6.):raise ValueError('Python NRM bounds differ')
        A=model.slopes.copy();D=model.intercepts.copy();frame(x.columns,A,D).to_csv(out/'starting_values.csv',index=False)
        r.update(fixture=str(fixture.relative_to(ROOT)),fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
            input_sha256={f:hashlib.sha256((pre/f).read_bytes()).hexdigest() for f in ('data.csv','integration.csv','evaluation_grid.csv')},
            settings=dict(model='baseline-category NRM',D=1,mean=0,variance=1,priors=None,GH=61,ML=True,
                max_iter=3000,R_parameter_tol=1e-8,python_loglik_tol=1e-10,python_item_ftol=1e-10,acceleration='none',SE=False),
            source_sha256={f:hashlib.sha256((ROOT/'validation'/f).read_bytes()).hexdigest() for f in ('nrm_r_comparison.py','nrm_r_comparison.R')})
        exe=rscript or shutil.which('Rscript')
        if not allow_fit or not exe:r['reason']='Actual R execution required';save();return r
        trace=subprocess.run(['git','show',HARNESS_COMMIT+':validation/r_mirt_phase1/optimizer_trace.R'],cwd=ROOT,capture_output=True,text=True,check=True)
        (out/'pr63_optimizer_trace.R').write_text(trace.stdout)
        r['trace_helper_sha256']=hashlib.sha256(trace.stdout.encode()).hexdigest()
        (out/'fit_started').write_text('One fit per engine; no retry\n')
        r['execution']['R']='EXECUTED'
        reference=run_r_process([exe,str(ROOT/'validation/nrm_r_comparison.R'),str(pre),str(out),'EXECUTE_NRM_ONCE'],out)
        r['execution']['R']='EXECUTED';r['fit_calls']['R']=int((out/'R_fit_invoked').exists())
        if reference.returncode:raise ValueError(f'R process exit {reference.returncode}; see preserved stderr')
        r['R']=pd.read_csv(out/'R_metadata.csv').iloc[0].to_dict()
        if str(r['R']['version'])!='1.48' or r['R']['mean']!=0 or r['R']['variance']!=1:raise ValueError('R metadata/specification mismatch')
        import mirt
        from mirt.estimation.em import EMEstimator,_PRECISE_ITEM_FTOL
        if _PRECISE_ITEM_FTOL!=1e-10:raise ValueError('Precise solver policy differs')
        r['execution']['python']='EXECUTED';r['fit_calls']['python']=1
        with (out/'Python_stdout.log').open('w') as stdout,(out/'Python_stderr.log').open('w') as stderr:
            with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
            fit=EMEstimator(n_quadpts=61,max_iter=3000,tol=1e-10,item_optim_ftol=_PRECISE_ITEM_FTOL,compute_standard_errors=False,item_priors=None,accelerate='none').fit(model,x.to_numpy(int),start=dict(slopes=A,intercepts=D))
        population=standard_normal_population(fit)
        rawA=fit.model.slopes.copy();rawD=fit.model.intercepts.copy();sgn=orientation(rawA,A)
        frame(x.columns,rawA,rawD).to_csv(out/'Python_raw_parameters.csv',index=False)
        r['python']=dict(converged=bool(fit.converged),iterations=int(fit.n_iterations),loglik=float(fit.log_likelihood),aic=float(fit.aic),bic=float(fit.bic),n_parameters=int(fit.n_parameters),orientation=sgn,**population)
        if fit.n_parameters!=48 or int(r['R']['n_parameters'])!=48:raise ValueError('NRM free-coordinate count differs')
        model=mirt.NominalResponseModel(8,4);model.set_parameters(slopes=sgn*rawA,intercepts=rawD)
        pp=frame(x.columns,model.slopes,model.intercepts);pp.to_csv(out/'Python_parameters.csv',index=False)
        rp=pd.read_csv(out/'R_parameters.csv').set_index('item').loc[x.columns].reset_index()
        for engine,p in [('python',pp),('R',rp)]:
            aa,dd=parameters(p)
            if not np.isfinite(np.c_[aa,dd]).all() or not np.array_equal(aa[:,0],np.zeros(8)) or not np.array_equal(dd[:,0],np.zeros(8)):raise ValueError('NRM baseline constraints differ')
            violation=float(max(0.,(abs(aa[:,1:])-5).max(),(abs(dd[:,1:])-6).max()))
            if violation:raise ValueError('NRM physical bound violation')
            ll,score=marginal(x.to_numpy(int),aa,dd,nodes,weights)
            r.setdefault('common_grid_loglik',{})[engine]=ll
            r.setdefault('scores',{})[engine]={k:float(abs(v[:,1:]).max()) for k,v in score.items()}
            r.setdefault('constraints',{})[engine]=dict(max_violation=violation,boundary=bool(np.any(abs(aa[:,1:])>=5-1e-6) or np.any(abs(dd[:,1:])>=6-1e-6)))
        r['fit_statistic_definitions']={}
        for engine,meta in [('python',r['python']),('R',r['R'])]:
            r['fit_statistic_definitions'][engine]=dict(N=500,k=48,
                native_minus_common_loglik=meta['loglik']-r['common_grid_loglik'][engine],
                AIC_identity_residual=meta['aic']-(-2*meta['loglik']+2*48),
                BIC_identity_residual=meta['bic']-(-2*meta['loglik']+np.log(500)*48))
            if not np.allclose([meta['aic'],meta['bic']],[-2*meta['loglik']+2*48,-2*meta['loglik']+np.log(500)*48],rtol=np.sqrt(np.finfo(float).eps),atol=np.sqrt(np.finfo(float).eps)):
                raise ValueError('NRM native IC definitions differ')
        def curves(theta):return fitted_category_curves(dict(result=type('View',(),{'model':model})(),k=8,items=x.columns,labels=labels),theta)
        evaluation=pd.read_csv(pre/'evaluation_grid.csv');ev=curves(evaluation.theta.to_numpy())
        ev['node']=np.tile(np.arange(len(evaluation)),8*4);ev['category']=ev.Kategori.map(dict(zip(labels,range(4))))
        ev=ev.rename(columns=dict(Madde='item',Theta='theta',Olasılık='probability'))
        info=np.column_stack([model.information(evaluation.theta.to_numpy(),item_idx=i) for i in range(8)])
        ev['information']=np.concatenate([np.tile(info[:,i],4) for i in range(8)])
        ev.to_csv(out/'Python_curves.csv',index=False)
        rc=pd.read_csv(out/'R_curves.csv');r['comparisons']=h.compare_curves(ev,rc)
        pair=ev.merge(rc,on=['item','node','category'],suffixes=('_python','_R'),validate='one_to_one')
        if len(pair)!=len(ev):raise ValueError('NRM curve keys differ')
        central=pair.theta_python.between(-4,4);ip=pair[central].drop_duplicates(['item','node'])
        tif=ip.groupby('node')[['information_python','information_R']].sum()
        if (tif<=0).any().any():raise ValueError('Undefined reporting-range SEM')
        sem_p,sem_r=1/np.sqrt(tif.information_python),1/np.sqrt(tif.information_R)
        pd.DataFrame(dict(node=tif.index,I_python=tif.information_python,I_R=tif.information_R,SEM_python=sem_p,SEM_R=sem_r)).to_csv(out/'information_SEM_comparison.csv',index=False)
        cp=pp.melt(id_vars='item',var_name='coefficient',value_name='Python').merge(rp.melt(id_vars='item',var_name='coefficient',value_name='R'),on=['item','coefficient'],validate='one_to_one')
        cp['absolute_difference']=abs(cp.Python-cp.R);den=abs(cp.R.to_numpy());cp['relative_difference']=np.divide(cp.absolute_difference.to_numpy(),den,out=np.full(len(cp),np.nan),where=den!=0)
        cp.to_csv(out/'parameter_comparison.csv',index=False)
        metrics=dict(parameters=float(cp.absolute_difference.max()),loglik=abs(r['common_grid_loglik']['python']-r['common_grid_loglik']['R']),
            probabilities=float(abs(pair.loc[central,'probability_python']-pair.loc[central,'probability_R']).max()),item_information=float(abs(ip.information_python-ip.information_R).max()),test_information=float(abs(tif.information_python-tif.information_R).max()),aic=abs(r['python']['aic']-r['R']['aic']),bic=abs(r['python']['bic']-r['R']['bic']))
        if not r['R']['converged'] or not fit.converged:raise ValueError('NRM backend NOT CONVERGED; person scores withheld')
        people=library_person_abilities(fit,x,model='NRM')
        people['Yetenek (θ)']*=sgn
        people.to_csv(out/'Python_persons.csv',index=False);rr=pd.read_csv(out/'R_persons.csv')
        if not np.array_equal(people.Birey,rr.Birey):raise ValueError('Person IDs/order differ')
        for key,col in [('EAP_theta','Yetenek (θ)'),('posterior_SD','SH (θ)')]:metrics[key]=float(abs(people[col]-rr[col]).max())
        people.merge(rr,on='Birey',suffixes=('_python','_R'),validate='one_to_one').to_csv(out/'person_comparison.csv',index=False)
        rtrace=pd.read_csv(out/'R_GH_curves.csv');rtrace['Theta']=nodes[rtrace.node.to_numpy(int)];rtrace['Kategori']=rtrace.category.map(dict(enumerate(labels)))
        replay=person_abilities(x,rtrace,model='NRM',category_labels=labels,quadrature=(nodes,weights))
        identity=float(abs(replay[['Yetenek (θ)','SH (θ)']].to_numpy()-rr[['Yetenek (θ)','SH (θ)']].to_numpy()).max())
        if identity>np.sqrt(np.finfo(float).eps):raise ValueError('Independent posterior identity failed')
        if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Nonfinite NRM metric')
        r.update(metrics=metrics,person_scoring_identity=identity,SEM_max_relative=float(abs(sem_p-sem_r).div(abs(sem_r)).max()),
            acceptance=acceptance(metrics,float(abs(sem_p-sem_r).div(abs(sem_r)).max()),8,criteria))
        r['numerical_budgets_passed']=all(v['passed'] for v in r['acceptance'].values())
        r.update(status='MEASURED / NOT CERTIFIED' if r['numerical_budgets_passed'] else 'FAIL',decision='MEASURED / NOT CERTIFIED' if r['numerical_budgets_passed'] else 'BLOCKED')
        if not r['numerical_budgets_passed']:r['reason']='Predeclared numerical budget exceeded; thresholds unchanged'
    except Exception as e:
        r['fit_calls']['R']=int((out/'R_fit_invoked').exists())
        if r['execution']['python']=='EXECUTED':
            import traceback
            (out/'Python_stderr.log').write_text(traceback.format_exc())
        r.update(status='FAIL',decision='BLOCKED',reason=f'{type(e).__name__}: {e}')
    save();return r
