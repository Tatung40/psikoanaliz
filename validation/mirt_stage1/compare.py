"""One preregistered Python/R pair per profile; retain failures, no retries."""
import argparse,itertools,json,subprocess,shutil,sys,hashlib,platform
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
import mirt
from mirt.estimation.quadrature import GaussHermiteQuadrature
from core.irt_multidim import fit,structure,STATUS,nominal_start
from validation.r_mirt_phase1.harness import discrepancies

ROOT=Path(__file__).resolve().parent
PHASE1=('2PL_2D','2PL_3D','GRM_2D','GRM_3D')
EXPANSION=('Rasch_2D','PCM_2D','GPCM_3D','GPCM_5D','NRM_2D','Bifactor_4D','2PL_EFA_2D','GRM_EFA_2D')
PROFILES=PHASE1
BINARY=('2PL','Rasch','Bifactor')


def profile(name):
 parts=name.split('_');model=parts[0];q=int(parts[-1][0]);exploratory='EFA' in parts
 k=4*q+2 if q<=3 else 3*q+2
 pattern=np.zeros((k,q),int)
 size=4 if q<=3 else 3
 for j in range(q):pattern[j*size:(j+1)*size,j]=1
 within=(model,q) in (('2PL',3),('GRM',2),('Rasch',2),('PCM',2),('NRM',2),('GPCM',5))
 if within:pattern[-2:,:]=1
 else:pattern[-2:,0]=1
 correlated=not within
 if model in ('Rasch','PCM'):correlated=False
 full=model in ('GPCM','NRM') and q<=3
 if full:correlated='full'
 groups=None
 if model=='Bifactor':
  groups=np.repeat(np.arange(q-1),3).tolist();k=len(groups)
  pattern=np.column_stack((np.ones(k,int),np.array(groups)[:,None]==np.arange(q-1)))
  correlated=False
 if exploratory:
  pattern[:,:]=1;pattern[:q,:]=np.tril(np.ones((q,q),int));correlated=False
 syntax=structure(pattern,correlated,exploratory) if model!='Bifactor' else '\n'.join(f'F{j+1} = '+','.join(str(i+1) for i in np.flatnonzero(pattern[:,j])) for j in range(q))
 return dict(name=name,model=model,q=q,k=k,pattern=pattern.astype(int).tolist(),correlated=correlated,
  covariance_mode='full' if full else 'pair',exploratory=exploratory,groups=groups,syntax=syntax,
  N=600 if model=='Bifactor' else 200 if q==5 else 400,
  seed=20261026+q+(model=='GRM')*10+(EXPANSION.index(name)+100 if name in EXPANSION else 0),
  n_quadpts=7 if model=='Bifactor' else 5 if q==5 else 15,max_iter=1000,tol=1e-7,
  criteria_sha256=hashlib.sha256((ROOT/'criteria.json').read_bytes()).hexdigest())


def masses(nodes,base,cov):
 log=np.log(base)+multivariate_normal.logpdf(nodes,cov=cov)-multivariate_normal.logpdf(nodes,cov=np.eye(nodes.shape[1]))
 return np.exp(log-logsumexp(log))


def likelihood(data,p,weight):
 if p.ndim==2:p=np.stack((1-p,p),-1)
 log=np.zeros((len(data),len(weight)))
 for j in range(data.shape[1]):log+=np.log(np.clip(p[:,j,:][:,data[:,j]].T,1e-300,1))
 return float(logsumexp(log+np.log(weight),axis=1).sum())


def inputs(name,out,refine_from=None):
 cfg=profile(name);out.mkdir(parents=True,exist_ok=True);q=cfg['q'];pattern=np.array(cfg['pattern'])
 from mirt.models.multidimensional import MultidimensionalModel
 from mirt.models.polytomous import GradedResponseModel,GeneralizedPartialCredit,NominalResponseModel
 from mirt.models.bifactor import BifactorModel
 model=cfg['model'];unit=model in ('Rasch','PCM')
 if model=='Bifactor':engine=BifactorModel(cfg['k'],cfg['groups'])
 elif model in BINARY:engine=MultidimensionalModel(cfg['k'],q,model_type='confirmatory',loading_pattern=pattern)
 elif model=='NRM':engine=NominalResponseModel(cfg['k'],4,n_factors=q)
 else:engine=(GradedResponseModel if model=='GRM' else GeneralizedPartialCredit)(cfg['k'],n_factors=q,n_categories=4)
 a=pattern*(1 if unit else 1.2)
 if cfg['exploratory']:a=pattern*(1+.3*np.sin((np.arange(cfg['k'])[:,None]+1)*(np.arange(q)[None,:]+1)))
 thresholds=np.tile(np.linspace(-1,1,3)*(0.4 if q==5 else 1),(cfg['k'],1));d=np.linspace(-.7,.7,cfg['k'])
 if model=='Bifactor':engine.set_parameters(general_loadings=np.linspace(.7,.95,cfg['k']),specific_loadings=np.tile([.8,.95,1.1],q-1),intercepts=d)
 elif model in BINARY:engine.set_parameters(slopes=a,intercepts=d)
 elif model=='NRM':engine.set_parameters(slopes=nominal_start(pattern,4),intercepts=np.tile([0,-.2,.2,.1],(cfg['k'],1)))
 else:engine.set_parameters(**{'discrimination':a,'thresholds' if model=='GRM' else 'steps':thresholds})
 covariance=np.eye(q)
 if cfg['correlated']:
  if cfg['correlated']=='full':covariance[:,:]=.25;np.fill_diagonal(covariance,1)
  else:covariance[0,1]=covariance[1,0]=.25
 theta=np.random.default_rng(cfg['seed']).multivariate_normal(np.zeros(q),covariance,cfg['N'])
 data=pd.DataFrame(mirt.simdata(model=engine,n_persons=cfg['N'],theta=theta,seed=cfg['seed']),columns=[f'M{i+1}' for i in range(cfg['k'])])
 data.to_csv(out/'data.csv',index=False)
 cfg['data_sha256']=hashlib.sha256((out/'data.csv').read_bytes()).hexdigest()
 if refine_from is not None:
  source=refine_from/name;old=json.loads((source/'report.json').read_text())
  if old['settings']['data_sha256']!=cfg['data_sha256'] or old['settings']['criteria_sha256']!=cfg['criteria_sha256']:raise ValueError('Refinement data/criteria mismatch')
  shutil.copy2(source/'report.json',out/'original_report.json')
  shutil.copy2(source/'python_parameters.npz',out/'python_start.npz')
  values=dict(np.load(out/'python_start.npz'));values.pop('covariance',None);engine.set_parameters(**values)
  cfg.update(max_iter=3000,tol=1e-9,initialization='Saved complete Python item solution; latent covariance reset to identity in BOTH engines',
   source_run=40394398251,start_sha256=hashlib.sha256((out/'python_start.npz').read_bytes()).hexdigest())
  a=engine.get_loading_matrix() if model=='Bifactor' else values['slopes' if model in BINARY or model=='NRM' else 'discrimination']
  if model=='NRM':
   table=pd.DataFrame(a.reshape(-1,q),columns=[f'a{j+1}' for j in range(q)]);table['d']=values['intercepts'].reshape(-1)
  else:
   table=pd.DataFrame(a,columns=[f'a{j+1}' for j in range(q)])
   if model in BINARY:table['d']=values['intercepts']
   else:
    d=-a.sum(1)[:,None]*values['thresholds' if model=='GRM' else 'steps']
    for c in range(3):table[f'd{c+1}']=d[:,c]
  table.to_csv(out/'start.csv',index=False,float_format='%.17g')
 (out/'settings.json').write_text(json.dumps(cfg,indent=2)+'\n')
 grid=GaussHermiteQuadrature(cfg['n_quadpts'],q)
 pd.DataFrame(np.column_stack((grid.nodes,grid.weights)),columns=[f'T{i+1}' for i in range(q)]+['weight']).to_csv(out/'integration.csv',index=False,float_format='%.17g')
 pd.DataFrame(list(itertools.product([-2.,0.,2.],repeat=q)),columns=[f'T{i+1}' for i in range(q)]).to_csv(out/'trace.csv',index=False)
 return cfg,data,engine,grid


def r_run(out,mode):
 target=out/mode;target.mkdir(exist_ok=True)
 with (target/'stdout.log').open('w') as stdout,(target/'stderr.log').open('w') as stderr:
  p=subprocess.run(['Rscript',str(ROOT/'reference.R'),str(out.resolve()),mode],stdout=stdout,stderr=stderr,timeout=1800)
 if p.returncode:raise RuntimeError(f'R {mode} failed (exit {p.returncode}); original stderr retained')
 return target


def native_outputs(engine,theta):
 p=np.asarray(engine.probability(theta))
 if p.ndim==2:p=np.stack((1-p,p),-1)
 info=np.stack([engine.item_information_matrix(theta,i) for i in range(engine.n_items)],1)
 return p,info


def arrays(directory,cfg):
 p=pd.read_csv(directory/'probabilities.csv').sort_values(['Node','Item','Category']).Probability.to_numpy().reshape(-1,cfg['k'],2 if cfg['model'] in BINARY else 4)
 info=pd.read_csv(directory/'information.csv').sort_values(['Node','Item','Row','Column']).Information.to_numpy().reshape(-1,cfg['k'],cfg['q'],cfg['q'])
 return p,info


def compare_pair(outcome,actual,cfg,data,grid,result,criteria):
 f=outcome['fit'];trace=pd.read_csv(actual.parent/'trace.csv').to_numpy()
 rmeta=json.loads((actual/'metadata.json').read_text());result['R']=rmeta
 rp,rI=arrays(actual,cfg);p,I=native_outputs(f.model,trace)
 pars=pd.read_csv(actual/'parameters.csv');a=pars.filter(regex='^a[1-5]$').to_numpy()
 d=pars.filter(regex='^d').to_numpy();params=f.model.parameters
 if cfg['model']=='NRM':
  pa=params['slopes'];pd_=params['intercepts'];a=a.reshape(cfg['k'],4,cfg['q']);d=d.reshape(cfg['k'],4)
 else:
  pa=f.model.get_loading_matrix() if cfg['model']=='Bifactor' else params['slopes' if cfg['model'] in BINARY else 'discrimination']
  key='thresholds' if cfg['model']=='GRM' else 'steps'
  pd_=params['intercepts'][:,None] if cfg['model'] in BINARY else -pa.sum(1)[:,None]*params[key]
 rcov=np.array(rmeta['covariance']);cov=outcome['covariance']
 rgrid=pd.read_csv(actual/'integration_probabilities.csv').sort_values(['Node','Item','Category']).Probability.to_numpy().reshape(len(grid.nodes),cfg['k'],-1)
 python_grid,_=native_outputs(f.model,grid.nodes)
 pll=likelihood(data.to_numpy(),python_grid,masses(grid.nodes,grid.weights,cov));rll=likelihood(data.to_numpy(),rgrid,masses(grid.nodes,grid.weights,rcov))
 score=mirt.fscores(f,data.to_numpy()[:16],method='EAP',n_quadpts=cfg['n_quadpts']);reap=pd.read_csv(actual/'eap.csv')
 checks=dict(slopes=discrepancies(pa,a,criteria['calibration_parameters']),intercepts=discrepancies(pd_,d,criteria['calibration_parameters']),
  covariance=discrepancies(cov,rcov,criteria['calibration_covariance']),probabilities=discrepancies(p,rp,criteria['calibration_probability']),
  information=discrepancies(I,rI,criteria['calibration_information']),EAP=discrepancies(score.theta,reap.filter(regex='^F').to_numpy(),criteria['calibration_EAP']),
  posterior_SD=discrepancies(score.standard_error,reap.filter(regex='^SD').to_numpy(),criteria['calibration_posterior_SD']),
  common_logLik=discrepancies([pll],[rll],criteria['common_logLik']),Python_native_LL=discrepancies([pll],[f.log_likelihood],criteria['native_common_logLik']),
  R_native_LL=discrepancies([rll],[rmeta['logLik']],criteria['native_common_logLik']))
 result['comparison']=checks;result['common_grid_logLik']=dict(Python=pll,R=rll)
 if not (f.converged and rmeta['converged']):result['status']='NOT CONVERGED'
 elif any(c['status']!='PASS' for c in checks.values()):result['status']='NUMERICAL DISAGREEMENT / NOT CERTIFIED'
 elif not rmeta['constraints_valid']:result['status']='NOT COMPARABLE'
 else:result['status']='NUMERICALLY VALIDATED / NOT CERTIFIED'
 return result


def run(name,out,refine_from=None):
 criteria=json.loads((ROOT/'criteria.json').read_text());cfg,data,engine,grid=inputs(name,out,refine_from)
 result=dict(profile=name,status='BLOCKED',R_execution='NOT EXECUTED',Python_execution='NOT EXECUTED',settings=cfg)
 try:
  if not shutil.which('Rscript'):raise RuntimeError('Independent R runtime unavailable')
  # Fixed-parameter objects, no parameter M-step: verify mapping before calibrating.
  result['R_execution']='PREFLIGHT ATTEMPTED'
  pre=r_run(out,'preflight');result['R_execution']='FIXED PARAMETER PREFLIGHT EXECUTED'
  unit=cfg['model'] in ('Rasch','PCM');start_a=np.array(cfg['pattern'])*(1 if unit else 1.2)
  if refine_from is None:
   if cfg['model']=='Bifactor':engine.set_parameters(general_loadings=np.full(cfg['k'],.7),specific_loadings=np.full(cfg['k'],.5),intercepts=np.zeros(cfg['k']))
   elif cfg['model'] in BINARY:engine.set_parameters(slopes=start_a,intercepts=np.zeros(cfg['k']))
   elif cfg['model']=='NRM':engine.set_parameters(slopes=nominal_start(cfg['pattern'],4),intercepts=np.zeros((cfg['k'],4)))
   else:engine.set_parameters(**{'discrimination':start_a,'thresholds' if cfg['model']=='GRM' else 'steps':np.tile(np.linspace(-1,1,3),(cfg['k'],1))})
  trace=pd.read_csv(out/'trace.csv').to_numpy();p,I=native_outputs(engine,trace);rp,rI=arrays(pre,cfg)
  cov=np.eye(cfg['q'])
  if cfg['correlated']:
   if cfg.get('covariance_mode')=='full':cov[:,:]=.25;np.fill_diagonal(cov,1)
   else:cov[0,1]=cov[1,0]=.25
  checks=dict(probabilities=discrepancies(p,rp,criteria['mapping_probability']),information=discrepancies(I,rI,criteria['mapping_information']),
   prior=discrepancies(masses(grid.nodes,grid.weights,cov),pd.read_csv(pre/'prior.csv').weight.to_numpy(),criteria['mapping_prior_mass']))
  result['preflight']=checks
  if any(c['status']!='PASS' for c in checks.values()):raise RuntimeError('NOT COMPARABLE: fixed-parameter mapping/weights failed; no calibration attempted')
  result['Python_execution']='CALIBRATION ATTEMPTED'
  outcome=fit(data,cfg['model'],cfg['groups'] if cfg['model']=='Bifactor' else cfg['pattern'],correlated=cfg['correlated'],exploratory=cfg.get('exploratory',False),labels=['0','1','2','3'] if cfg['model']=='NRM' else None,start_values={k:v for k,v in np.load(out/'python_start.npz').items() if k!='covariance'} if refine_from else None,n_quadpts=cfg['n_quadpts'],max_iter=cfg['max_iter'],tol=cfg['tol'])
  f=outcome['fit'];result['Python_execution']='EXECUTED'
  np.savez(out/'python_parameters.npz',**f.model.parameters,covariance=outcome['covariance'])
  py=dict(engine='Python mirt '+mirt.__version__,python=platform.python_version(),converged=bool(f.converged),logLik=f.log_likelihood,AIC=f.aic,BIC=f.bic,n_parameters=f.n_parameters,iterations=f.n_iterations)
  (out/'python_metadata.json').write_text(json.dumps(py,indent=2)+'\n');result['Python']=py
  result['R_execution']='CALIBRATION ATTEMPTED'
  actual=r_run(out,'fit');result['R_execution']='CALIBRATION EXECUTED'
  result=compare_pair(outcome,actual,cfg,data,grid,result,criteria)
 except Exception as e:result['blocker']=str(e)
 (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 return result

def recover(name,prior,out):
 """No new calibration: reevaluate retained native R estimates at full precision."""
 source=prior/name;original=json.loads((source/'report.json').read_text())
 cfg=original['settings'];criteria=json.loads((ROOT/'criteria.json').read_text())
 if cfg['criteria_sha256']!=hashlib.sha256((ROOT/'criteria.json').read_bytes()).hexdigest():
  raise ValueError('Cannot reuse evidence after criterion changes')
 if not (original['Python']['converged'] and original['R']['converged']):raise ValueError('Only converged retained fits can be reused')
 if any(v['status']!='PASS' for k,v in original['comparison'].items() if k!='R_native_LL'):
  raise ValueError('Original discrepancies other than rounded R metadata remain')
 if hashlib.sha256((source/'data.csv').read_bytes()).hexdigest()!=cfg['data_sha256']:
  raise ValueError('Original data hash mismatch')
 shutil.copytree(source,out,dirs_exist_ok=True)
 shutil.copy2(out/'report.json',out/'previous_report.json')
 if not (out/'original_report.json').exists():shutil.copy2(out/'report.json',out/'original_report.json')
 result=original.copy();result['provenance']='Retained converged calibration; only native saved-parameter likelihood reevaluated, NO M-STEP'
 try:
  target=r_run(out,'evaluate_saved');precise=json.loads((target/'metadata.json').read_text())
  check=discrepancies([original['common_grid_logLik']['R']],[precise['logLik']],criteria['native_common_logLik'])
  result['comparison']=dict(original['comparison'])
  result['comparison']['R_native_LL']=dict(check,source='Saved native estimates, fixed-parameter R evaluation; original rounded field retained in original_report.json')
  result['R_saved_estimate_evaluation']=precise
  result['status']='NUMERICALLY VALIDATED / NOT CERTIFIED' if check['status']=='PASS' and precise['constraints_valid'] else 'NUMERICAL DISAGREEMENT / NOT CERTIFIED'
 except Exception as e:result['status']='BLOCKED';result['blocker']=str(e)
 (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 return result


def precision(name,prior,out):
 """One public-library Python continuation; retain the actual converged R fit."""
 if name not in ('Bifactor_4D','GRM_EFA_2D'):raise ValueError('Undiagnosed precision continuation prohibited')
 source=prior/name;old=json.loads((source/'report.json').read_text());cfg=dict(old['settings'])
 criteria=json.loads((ROOT/'criteria.json').read_text())
 if cfg['criteria_sha256']!=hashlib.sha256((ROOT/'criteria.json').read_bytes()).hexdigest() or cfg['data_sha256']!=hashlib.sha256((source/'data.csv').read_bytes()).hexdigest():raise ValueError('Data/criteria changed')
 if not (old['R']['converged'] and old['Python']['converged']):raise ValueError('Stationarity continuation requires retained converged fits')
 shutil.copytree(source,out,dirs_exist_ok=True);shutil.copy2(out/'report.json',out/'precision_previous_report.json');shutil.copy2(out/'settings.json',out/'R_source_settings.json')
 cfg.update(tol=1e-11,max_iter=3000,source_run=40399106868,initialization='Own saved Python item solution, no R parameters copied',
  precision='Public EM item_optim_ftol=1e-14 for GRM; public bfactor M-step controls unchanged')
 (out/'settings.json').write_text(json.dumps(cfg,indent=2)+'\n')
 result=dict(profile=name,status='BLOCKED',settings=cfg,preflight=old['preflight'],
  preflight_provenance='Retained actual R mapping at original documented common starts; no new preflight claimed',
  Python_execution='CALIBRATION ATTEMPTED',R_execution='RETAINED INDEPENDENT CONVERGED CALIBRATION / NO NEW R FIT')
 try:
  data=pd.read_csv(out/'data.csv');start={k:v for k,v in np.load(out/'python_parameters.npz').items() if k!='covariance'}
  if cfg['model']=='Bifactor':outcome=fit(data,'Bifactor',cfg['groups'],n_quadpts=cfg['n_quadpts'],max_iter=3000,tol=1e-11,start_values=start)
  else:
   from mirt.models.polytomous import GradedResponseModel
   engine=GradedResponseModel(cfg['k'],4,n_factors=cfg['q'],item_names=list(data.columns))
   engine.set_free_parameter_masks(dict(discrimination=np.array(cfg['pattern'],bool),thresholds=np.ones((cfg['k'],3),bool)))
   estimator=mirt.EMEstimator(n_quadpts=cfg['n_quadpts'],max_iter=3000,tol=1e-11,item_optim_ftol=1e-14,item_optim_maxiter=200,compute_standard_errors=False,accelerate='none')
   f=estimator.fit(engine,data.to_numpy(),start=start)
   a=f.model.parameters['discrimination'];pattern=np.array(cfg['pattern'],bool)
   if np.any(a[~pattern]!=0) or np.any(a[pattern]<.1) or np.any(a[pattern]>5):raise ValueError('Precision fit violated loading constraints')
   b=f.model.parameters['thresholds']
   if np.any(np.abs(b)>6+1e-12) or np.any(np.diff(b,axis=1)<1e-6-1e-12):raise ValueError('Precision fit violated threshold constraints')
   outcome=dict(fit=f,covariance=np.eye(cfg['q']))
  f=outcome['fit'];result['Python_execution']='EXECUTED'
  np.savez(out/'python_parameters.npz',**f.model.parameters,covariance=outcome['covariance'])
  result['Python']=dict(engine='Python mirt '+mirt.__version__,python=platform.python_version(),converged=bool(f.converged),logLik=f.log_likelihood,AIC=f.aic,BIC=f.bic,n_parameters=f.n_parameters,iterations=f.n_iterations)
  (out/'python_metadata.json').write_text(json.dumps(result['Python'],indent=2)+'\n')
  grid=GaussHermiteQuadrature(cfg['n_quadpts'],cfg['q'])
  result=compare_pair(outcome,out/'fit',cfg,data,grid,result,criteria)
 except Exception as e:result['blocker']=str(e)
 (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 return result


def reuse_accepted(names,prior,out):
 """Fail closed on missing/rejected baseline evidence; NEVER fallback to fits."""
 for name in names:
  record=json.loads((prior/name/'report.json').read_text())
  if record['status']!='NUMERICALLY VALIDATED / NOT CERTIFIED':raise ValueError('Baseline reuse requires accepted evidence: '+name)
 return [recover(name,prior,out/name) for name in names]


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--reuse',type=Path);parser.add_argument('--expand',action='store_true');parser.add_argument('--refine-failed',action='store_true');parser.add_argument('--python-precision',action='store_true');args=parser.parse_args()
 names=PHASE1+EXPANSION if args.expand else PHASE1
 reports=reuse_accepted(names,args.reuse,args.output) if args.reuse and not (args.python_precision or args.refine_failed) else [recover(name,args.reuse,args.output/name) if args.reuse and (args.reuse/name/'report.json').exists() and json.loads((args.reuse/name/'report.json').read_text())['status']=='NUMERICALLY VALIDATED / NOT CERTIFIED' else precision(name,args.reuse,args.output/name) if args.python_precision else run(name,args.output/name,args.reuse if args.refine_failed else None) for name in names]
 print(json.dumps([dict(profile=r['profile'],status=r['status'],blocker=r.get('blocker')) for r in reports],indent=2))
 sys.exit(0 if all(r['status']=='NUMERICALLY VALIDATED / NOT CERTIFIED' for r in reports) else 1)
