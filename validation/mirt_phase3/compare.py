"""Opt-in independent R calibration; preserve failures, one pair/case, no retry."""
import argparse,hashlib,json,platform,shutil,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.optimize._numdiff import approx_derivative
from validation.mirt_phase3 import engine
from validation.mirt_stage1.compare import likelihood
from validation.r_mirt_phase1.harness import discrepancies

ROOT=Path(__file__).resolve().parent;CRITERIA=ROOT.parent/'mirt_stage1/criteria.json'


def hessian(obj,x,bounds):
 # Interior tangent coordinates only: active box coordinates excluded explicitly.
 box=np.asarray(bounds);eps=np.sqrt(np.finfo(float).eps)*np.maximum(1,abs(x));free=(x-box[:,0]>eps)&(box[:,1]-x>eps);idx=np.flatnonzero(free)
 matrices=[]
 for step in (1e-4,5e-5):
  def gradient(v):
   z=x.copy();z[idx]=v;return obj.evaluate(z)[1][idx]
  H=approx_derivative(gradient,x[idx],method='3-point',abs_step=step,bounds=(box[idx,0],box[idx,1]));matrices.append((H+H.T)/2)
 eigen=np.linalg.eigvalsh(matrices[1]);relative=float(np.linalg.norm(matrices[1]-matrices[0])/max(1,np.linalg.norm(matrices[1])))
 return dict(steps=[1e-4,5e-5],free_coordinates=idx.tolist(),excluded_active_coordinates=np.flatnonzero(~free).tolist(),smallest_eigenvalues=eigen[:5].tolist(),condition=float(np.max(abs(eigen))/max(np.min(abs(eigen)),np.finfo(float).tiny)),relative_step_difference=relative,
  interpretation='Diagnostic only: no SE/CI, global optimality or structural identification certificate')


def r_run(root,mode):
 target=root/mode;target.mkdir(exist_ok=True)
 with (target/'stdout.log').open('w') as stdout,(target/'stderr.log').open('w') as stderr:
  run=subprocess.run(['Rscript',str(ROOT/'reference.R'),str(root.resolve()),mode],stdout=stdout,stderr=stderr,timeout=1800)
 if run.returncode:raise RuntimeError(f'R {mode} error {run.returncode}; original stderr preserved')
 return target


def pack_reference(spec,folder):
 pars=pd.read_csv(folder/'parameters.csv');meta=json.loads((folder/'metadata.json').read_text())
 A=pars.filter(regex='^a[1-3]$').to_numpy();parts=[A.ravel()[spec.active],pars.delta.to_numpy()]
 if spec.free_c:parts.append(pars.c.to_numpy())
 if spec.free_u:parts.append(pars.u.to_numpy())
 if spec.correlated:parts.append(np.atleast_1d(meta['eta']))
 return np.concatenate(parts),meta


def run_case(family,q,free_c,free_u,out):
 out.mkdir(parents=True,exist_ok=True);data=pd.read_csv(ROOT/f'fixtures/2PL_{q}D.csv');J=len(data.columns)
 pattern=np.zeros((J,q),int)
 for d in range(q):pattern[d*4:(d+1)*4,d]=1
 pattern[-2:]=np.ones(q,int) if q==3 else np.array([1,0])
 spec=engine.Specification(pattern,family,True,free_c,free_u,c=.2,u=.95 if family=='4PL' else 1.)
 obj=engine.Objective(data,spec,15);start,bounds=spec.start_bounds();criteria=json.loads(CRITERIA.read_text())
 data.to_csv(out/'data.csv',index=False);theta=np.vstack([np.zeros(q),np.eye(q),-np.eye(q),np.linspace(-3,3,31)[:,None]*np.ones(q)])
 pd.DataFrame(theta,columns=[f'F{d+1}' for d in range(q)]).to_csv(out/'trace.csv',index=False)
 integration=pd.DataFrame(obj.grid.nodes,columns=[f'F{d+1}' for d in range(q)]);integration['weight']=obj.grid.weights;integration.to_csv(out/'integration.csv',index=False,float_format='%.17g')
 settings=dict(family=family,q=q,pattern=pattern.tolist(),correlated=True,c=spec.c,u=spec.u,free_c=free_c,free_u=free_u,
  Python=platform.python_version(),scipy=scipy.__version__,engine='NEW experimental SciPy component, NOT Python mirt native calibration',
  N=len(data),J=J,n_quadpts=15,seed=20261027,priors=None,identification='Mean0, unit variances, >=3 positive pure anchors/axis',
  correlation='Full normalized-Cholesky eta, lower-triangular row order; eta in [-3,3]',start=start.tolist(),bounds=bounds,
  Python_stopping='L-BFGS-B ftol1e-13 / gtol1e-7 / max2000; actual message and score retained',R_stopping='mirt EM max parameter change <1e-8, max3000, SLSQP xtol1e-8',
  criteria_sha256=hashlib.sha256(CRITERIA.read_bytes()).hexdigest(),data_sha256=hashlib.sha256((out/'data.csv').read_bytes()).hexdigest(),
  source='Retained PR80 2PL fixture; no new simulated responses; asymptote-model misspecification is not a recovery study')
 (out/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
 result=dict(status='BLOCKED',settings=settings,R_execution='NOT EXECUTED',Python_execution='NOT EXECUTED',certification='NOT CERTIFIED')
 try:
  if not shutil.which('Rscript'):raise RuntimeError('R VALIDATION BLOCKED: Rscript unavailable')
  pre=r_run(out,'preflight');result['R_execution']='FIXED PARAMETER PREFLIGHT EXECUTED';px=start.copy();px[-spec.n_cov:]=.2
  rp=pd.read_csv(pre/'probabilities.csv').sort_values(['Node','Item']).Probability.to_numpy().reshape(len(theta),J)
  ri=pd.read_csv(pre/'information.csv').sort_values(['Node','Item','Row','Column']).Information.to_numpy().reshape(len(theta),J,q,q)
  checks=dict(probability=discrepancies(engine.probabilities(spec,px,theta),rp,criteria['mapping_probability']),information=discrepancies(engine.information(spec,px,theta),ri,criteria['mapping_information']),prior=discrepancies(obj.masses(px[-spec.n_cov:]),pd.read_csv(pre/'prior.csv').weight.to_numpy(),criteria['mapping_prior_mass']))
  result['preflight']=checks
  if any(v['status']!='PASS' for v in checks.values()):raise ValueError('NOT COMPARABLE: preflight mismatch; no fits')
  py=engine.fit(data,spec,n_quadpts=15,tol=1e-13);x=py['parameters'];result['Python_execution']='ONE ACTUAL SCIPY MML FIT EXECUTED'
  np.save(out/'python_parameters.npy',x);result['Python']={k:v for k,v in py.items() if k not in ('parameters','specification','objective')}
  (out/'python_metadata.json').write_text(json.dumps(result['Python'],indent=2)+'\n')
  result['R_execution']='CALIBRATION ATTEMPTED';actual=r_run(out,'fit');rx,r=pack_reference(spec,actual);result['R_execution']='ONE ACTUAL INDEPENDENT R MIRT FIT EXECUTED';result['R']=r
  pp=engine.probabilities(spec,x,theta);pI=engine.information(spec,x,theta)
  rp=pd.read_csv(actual/'probabilities.csv').sort_values(['Node','Item']).Probability.to_numpy().reshape(len(theta),J)
  rI=pd.read_csv(actual/'information.csv').sort_values(['Node','Item','Row','Column']).Information.to_numpy().reshape(len(theta),J,q,q)
  rgrid=pd.read_csv(actual/'integration_probabilities.csv').sort_values(['Node','Item']).Probability.to_numpy().reshape(len(obj.grid.nodes),J)
  rw=pd.read_csv(actual/'prior.csv').weight.to_numpy();pll=-obj.evaluate(x)[0];rll=likelihood(data.to_numpy(),rgrid,rw)
  mu,pcov=engine.eap(obj,x);ell=data.to_numpy()@np.log(rgrid).T+(1-data.to_numpy())@np.log1p(-rgrid).T+np.log(rw);ell-=__import__('scipy').special.logsumexp(ell,axis=1)[:,None];posterior=np.exp(ell)
  rmu=posterior@obj.grid.nodes;rcov=np.einsum('nk,kr,ks->nrs',posterior,obj.grid.nodes,obj.grid.nodes)-np.einsum('nr,ns->nrs',rmu,rmu)
  checks=dict(parameters=discrepancies(x,rx,criteria['calibration_parameters']),covariance=discrepancies(spec.covariance(spec.unpack(x)[-1]),r['covariance'],criteria['calibration_covariance']),
   probabilities=discrepancies(pp,rp,criteria['calibration_probability']),information=discrepancies(pI,rI,criteria['calibration_information']),TIF=discrepancies(pI.sum(1),rI.sum(1),criteria['calibration_information']),
   EAP=discrepancies(mu,rmu,criteria['calibration_EAP']),posterior_SD=discrepancies(np.sqrt(np.diagonal(pcov,axis1=1,axis2=2)),np.sqrt(np.diagonal(rcov,axis1=1,axis2=2)),criteria['calibration_posterior_SD']),
   common_logLik=discrepancies([pll],[rll],criteria['common_logLik']),R_native_LL=discrepancies([rll],[r['logLik']],criteria['native_common_logLik']),Python_native_LL=discrepancies([pll],[py['logLik']],criteria['native_common_logLik']),
   AIC=discrepancies([py['AIC']],[r['AIC']],criteria['common_logLik']),BIC=discrepancies([py['BIC']],[r['BIC']],criteria['common_logLik']))
  result['comparison']=checks;result['common_grid_logLik']=dict(Python=pll,R=rll)
  result['Hessian']=dict(Python=hessian(obj,x,bounds),R=hessian(obj,rx,bounds));result['R_physical_score']=obj.evaluate(rx)[1].tolist()
  result['status']='NUMERICALLY VALIDATED / NOT CERTIFIED' if py['converged'] and r['converged'] and r['constraints_valid'] and all(v['status']=='PASS' for v in checks.values()) else 'NOT CONVERGED' if not (py['converged'] and r['converged']) else 'NUMERICAL DISAGREEMENT / NOT CERTIFIED'
 except Exception as exc:result['blocker']=str(exc)
 (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result


def execute(request,out):
 cfg=json.loads(Path(request).read_text());reports=[]
 for case in cfg['cases']:
  result=run_case(case['family'],case['q'],case['free_c'],case['free_u'],out/case['name'])
  reports.append(result)
  if result['status']!='NUMERICALLY VALIDATED / NOT CERTIFIED':break
 record=dict(status='NUMERICALLY VALIDATED / NOT CERTIFIED' if reports and len(reports)==len(cfg['cases']) and all(r['status']=='NUMERICALLY VALIDATED / NOT CERTIFIED' for r in reports) else 'FAIL' if reports else 'NOT REQUESTED',cases=reports)
 out.mkdir(parents=True,exist_ok=True);(out/'report.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');return record


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,default=ROOT/'request.json');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();r=execute(args.request,args.output)
 print('PHASE3_REPORT='+json.dumps(r));raise SystemExit(0 if r['status'] in ('NUMERICALLY VALIDATED / NOT CERTIFIED','NOT REQUESTED') else 1)
