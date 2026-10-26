"""One preregistered Python/R pair per profile; retain failures, no retries."""
import argparse,itertools,json,subprocess,shutil,sys,hashlib,platform
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
import mirt
from mirt.estimation.quadrature import GaussHermiteQuadrature
from core.irt_multidim import fit,structure,STATUS
from validation.r_mirt_phase1.harness import discrepancies

ROOT=Path(__file__).resolve().parent
PROFILES=('2PL_2D','2PL_3D','GRM_2D','GRM_3D')


def profile(name):
 model,dim=name.split('_');q=int(dim[0]);k=4*q+2
 pattern=np.zeros((k,q),int)
 for j in range(q):pattern[j*4:(j+1)*4,j]=1
 within=(model,q) in (('2PL',3),('GRM',2))
 if within:pattern[-2:,:]=1
 else:
  pattern[-2:,0]=1
 correlated=not within
 return dict(name=name,model=model,q=q,k=k,pattern=pattern.tolist(),correlated=correlated,
  syntax=structure(pattern,correlated),N=400,seed=20261026+q+(model=='GRM')*10,
  n_quadpts=15,max_iter=1000,tol=1e-7,criteria_sha256=hashlib.sha256((ROOT/'criteria.json').read_bytes()).hexdigest())


def masses(nodes,base,cov):
 log=np.log(base)+multivariate_normal.logpdf(nodes,cov=cov)-multivariate_normal.logpdf(nodes,cov=np.eye(nodes.shape[1]))
 return np.exp(log-logsumexp(log))


def likelihood(data,p,weight):
 if p.ndim==2:p=np.stack((1-p,p),-1)
 log=np.zeros((len(data),len(weight)))
 for j in range(data.shape[1]):log+=np.log(np.clip(p[:,j,:][:,data[:,j]].T,1e-300,1))
 return float(logsumexp(log+np.log(weight),axis=1).sum())


def inputs(name,out):
 cfg=profile(name);out.mkdir(parents=True,exist_ok=True);q=cfg['q'];pattern=np.array(cfg['pattern'])
 from mirt.models.multidimensional import MultidimensionalModel
 from mirt.models.polytomous import GradedResponseModel
 engine=MultidimensionalModel(cfg['k'],q,model_type='confirmatory',loading_pattern=pattern) if cfg['model']=='2PL' else GradedResponseModel(cfg['k'],n_factors=q,n_categories=4)
 a=pattern*1.2;thresholds=np.tile(np.linspace(-1,1,3),(cfg['k'],1));d=np.linspace(-.7,.7,cfg['k'])
 if cfg['model']=='2PL':engine.set_parameters(slopes=a,intercepts=d)
 else:engine.set_parameters(discrimination=a,thresholds=thresholds)
 covariance=np.eye(q)
 if cfg['correlated']:covariance[0,1]=covariance[1,0]=.25
 theta=np.random.default_rng(cfg['seed']).multivariate_normal(np.zeros(q),covariance,cfg['N'])
 data=pd.DataFrame(mirt.simdata(model=engine,n_persons=cfg['N'],theta=theta,seed=cfg['seed']),columns=[f'M{i+1}' for i in range(cfg['k'])])
 data.to_csv(out/'data.csv',index=False)
 cfg['data_sha256']=hashlib.sha256((out/'data.csv').read_bytes()).hexdigest()
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
 p=pd.read_csv(directory/'probabilities.csv').sort_values(['Node','Item','Category']).Probability.to_numpy().reshape(-1,cfg['k'],2 if cfg['model']=='2PL' else 4)
 info=pd.read_csv(directory/'information.csv').sort_values(['Node','Item','Row','Column']).Information.to_numpy().reshape(-1,cfg['k'],cfg['q'],cfg['q'])
 return p,info


def run(name,out):
 criteria=json.loads((ROOT/'criteria.json').read_text());cfg,data,engine,grid=inputs(name,out)
 result=dict(profile=name,status='BLOCKED',R_execution='NOT EXECUTED',Python_execution='NOT EXECUTED',settings=cfg)
 try:
  if not shutil.which('Rscript'):raise RuntimeError('Independent R runtime unavailable')
  # Fixed-parameter objects, no parameter M-step: verify mapping before calibrating.
  result['R_execution']='PREFLIGHT ATTEMPTED'
  pre=r_run(out,'preflight');result['R_execution']='FIXED PARAMETER PREFLIGHT EXECUTED'
  start_a=np.array(cfg['pattern'])*1.2
  if cfg['model']=='2PL':engine.set_parameters(slopes=start_a,intercepts=np.zeros(cfg['k']))
  else:engine.set_parameters(discrimination=start_a,thresholds=np.tile(np.linspace(-1,1,3),(cfg['k'],1)))
  trace=pd.read_csv(out/'trace.csv').to_numpy();p,I=native_outputs(engine,trace);rp,rI=arrays(pre,cfg)
  cov=np.eye(cfg['q'])
  if cfg['correlated']:cov[0,1]=cov[1,0]=.25
  checks=dict(probabilities=discrepancies(p,rp,criteria['mapping_probability']),information=discrepancies(I,rI,criteria['mapping_information']),
   prior=discrepancies(masses(grid.nodes,grid.weights,cov),pd.read_csv(pre/'prior.csv').weight.to_numpy(),criteria['mapping_prior_mass']))
  result['preflight']=checks
  if any(c['status']!='PASS' for c in checks.values()):raise RuntimeError('NOT COMPARABLE: fixed-parameter mapping/weights failed; no calibration attempted')
  result['Python_execution']='CALIBRATION ATTEMPTED'
  outcome=fit(data,cfg['model'],cfg['pattern'],correlated=cfg['correlated'],n_quadpts=cfg['n_quadpts'],max_iter=cfg['max_iter'],tol=cfg['tol'])
  f=outcome['fit'];result['Python_execution']='EXECUTED'
  np.savez(out/'python_parameters.npz',**f.model.parameters,covariance=outcome['covariance'])
  py=dict(engine='Python mirt '+mirt.__version__,python=platform.python_version(),converged=bool(f.converged),logLik=f.log_likelihood,AIC=f.aic,BIC=f.bic,n_parameters=f.n_parameters,iterations=f.n_iterations)
  (out/'python_metadata.json').write_text(json.dumps(py,indent=2)+'\n');result['Python']=py
  result['R_execution']='CALIBRATION ATTEMPTED'
  actual=r_run(out,'fit');result['R_execution']='CALIBRATION EXECUTED'
  rmeta=json.loads((actual/'metadata.json').read_text());result['R']=rmeta
  rp,rI=arrays(actual,cfg);p,I=native_outputs(f.model,trace)
  pars=pd.read_csv(actual/'parameters.csv');a=pars.filter(regex='^a[1-3]$').to_numpy()
  d=pars.filter(regex='^d').to_numpy();params=f.model.parameters
  pa=params['slopes' if cfg['model']=='2PL' else 'discrimination'];pd_=params['intercepts'][:,None] if cfg['model']=='2PL' else -pa.sum(1)[:,None]*params['thresholds']
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
 except Exception as e:result['blocker']=str(e)
 (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 return result

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 reports=[run(name,args.output/name) for name in PROFILES]
 print(json.dumps([dict(profile=r['profile'],status=r['status'],blocker=r.get('blocker')) for r in reports],indent=2))
 sys.exit(0 if all(r['status']=='NUMERICALLY VALIDATED / NOT CERTIFIED' for r in reports) else 1)
