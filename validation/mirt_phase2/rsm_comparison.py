"""Exact orthogonal Andrich RSM; ONE genuine joint shared-step fit per engine/case."""
import hashlib,json,platform,shutil,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import mirt
from mirt.models.polytomous import RatingScaleModel
from mirt.estimation.quadrature import GaussHermiteQuadrature
from core import irt_multidim_rsm_stack as engine
from validation.mirt_phase2.factorized_comparison import CRITERIA,ROOT
from validation.mirt_stage1.compare import likelihood
from validation.r_mirt_phase1.harness import discrepancies


def dataset(q,n):
 source=Path('validation/evidence/p5_browser/poly_ordinal.csv');raw=pd.read_csv(source);rng=np.random.default_rng(20261027);parts=[];maps=[]
 for d in range(q):
  order=rng.permutation(np.arange(d*n,(d+1)*n));part=raw.iloc[order,:6].reset_index(drop=True).add_prefix(f'F{d+1}_');parts.append(part);maps.append(order.tolist())
 data=pd.concat(parts,axis=1).astype(int);p=np.zeros((6*q,q),int)
 for d in range(q):p[d*6:(d+1)*6,d]=1
 return data,p,dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),seed=20261027,row_maps=maps,kind='Composite respondents; disjoint existing row ranges, original categories; no simulated responses')


def r_run(root,mode):
 out=root/mode;out.mkdir(exist_ok=True)
 with (out/'stdout.log').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
  run=subprocess.run(['Rscript',str(ROOT/'rsm_reference.R'),str(root.resolve()),mode],stdout=stdout,stderr=stderr,timeout=1800)
 if run.returncode:raise RuntimeError(f'R RSM {mode} failed ({run.returncode}); original stderr retained')
 return out


def native_information(model,theta):return np.column_stack([model.information(theta,item_idx=j) for j in range(model.n_items)])


def common_ll(model,data,p,grid):
 return sum(likelihood(data.to_numpy()[:,p[:,d].astype(bool)],model.probability(grid.nodes)[:,p[:,d].astype(bool),:],grid.weights) for d in range(p.shape[1]))


def score_diagnostic(params,data,p,grid):
 # Common physical coordinates: beta_j and tau_2...; tau_1=0 fixed.
 vector=np.r_[params['difficulty'],params['thresholds'][1:]];j=len(data.columns);values=[]
 model=RatingScaleModel(j,len(params['thresholds'])+1)
 for step in (1e-4,5e-5):
  score=[]
  for k in range(len(vector)):
   pair=[]
   for sign in (1,-1):
    v=vector.copy();v[k]+=sign*step;model.set_parameters(difficulty=v[:j],thresholds=np.r_[0.,v[j:]])
    pair.append(common_ll(model,data,p,grid))
   score.append((pair[0]-pair[1])/(2*step))
  values.append(dict(step=step,physical_score=score,max_absolute=float(np.max(np.abs(score)))))
 return dict(status='DIAGNOSTIC ONLY; not a convergence/SE certification',steps=values)



def restore_saved_python(values,metadata,data,pattern):
 """Public fixed-parameter reevaluation only; never update an accepted estimate."""
 model=RatingScaleModel(len(data.columns),len(values['thresholds'])+1,item_names=list(data));model.set_parameters(**values);masks=model.free_parameter_masks
 model.set_free_parameter_masks({key:np.zeros_like(mask,dtype=bool) for key,mask in masks.items()})
 evaluated=mirt.EMEstimator(n_quadpts=61,max_iter=1,tol=1e-10,compute_standard_errors=False,accelerate='none').fit(model,engine.stack_responses(data,pattern),start=values)
 for key in values:np.testing.assert_array_equal(model.parameters[key],values[key])
 if abs(evaluated.log_likelihood-metadata['logLik'])>json.loads(CRITERIA.read_text())['native_common_logLik']:raise ValueError('Saved Python likelihood mismatch')
 model.set_free_parameter_masks(masks)
 return mirt.FitResult(model=model,log_likelihood=metadata['logLik'],n_iterations=metadata['iterations'],converged=metadata['converged'],standard_errors={},aic=metadata['AIC'],bic=metadata['Native_stacked_BIC'],n_observations=len(data)*pattern.shape[1],n_parameters=metadata['free_parameters'],latent_mean=np.zeros(1),latent_covariance=np.eye(1))

def execute_case(q,n,root,prior=None):
 root.mkdir(parents=True,exist_ok=True);data,p,provenance=dataset(q,n);C=int(data.max().max())+1;J=len(p);grid=GaussHermiteQuadrature(61,1);theta=np.linspace(-4,4,81)[:,None]
 criteria=json.loads(CRITERIA.read_text());settings=dict(family='RSM',dimensions=q,N=n,categories=C,items=J,stacked_N=n*q,D=1,latent='Independent N(0,1)',mean=0,variance=1,priors=None,
  n_quadpts=61,Python='mirt '+mirt.__version__,python=platform.python_version(),max_iter=1000,tol=1e-10,item_optim_ftol=1e-13,item_optim_maxiter=200,acceleration='none',
  start='All beta=0, global tau=(0,1,...,C-2), first tau fixed0',bounds=dict(beta=[-6,6],free_tau=[-6,6]),shared_steps='ONE globally estimated vector; no ordering constraints',
  identification='Fixed unit between-item loadings, orthogonal named dimensions; first shared step fixed0',
  R_stopping='Maximum EM parameter change <1e-8, max3000; L-BFGS-B, no acceleration',
  BIC_definition='-2 joint LL + (J+C-2)*log(original N); native stacked BIC retained separately',
  criteria_sha256=hashlib.sha256(CRITERIA.read_bytes()).hexdigest(),dataset=provenance)
 data.to_csv(root/'data.csv',index=False);pd.DataFrame(p,columns=[f'F{d+1}' for d in range(q)]).to_csv(root/'pattern.csv',index=False)
 stacked=engine.stack_responses(data,p).astype(float);stacked[stacked<0]=np.nan;pd.DataFrame(stacked,columns=data.columns).to_csv(root/'stacked.csv',index=False,na_rep='NA')
 pd.DataFrame(dict(theta=grid.nodes[:,0],weight=grid.weights)).to_csv(root/'integration.csv',index=False,float_format='%.17g')
 settings['data_sha256']=hashlib.sha256((root/'data.csv').read_bytes()).hexdigest();settings['stacked_sha256']=hashlib.sha256((root/'stacked.csv').read_bytes()).hexdigest()
 (root/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
 result=dict(status='BLOCKED',settings=settings,R_execution='NOT EXECUTED',Python_execution='NOT EXECUTED',certification='NOT CERTIFIED')
 try:
  if not shutil.which('Rscript'):raise RuntimeError('R VALIDATION BLOCKED: Rscript unavailable')
  fixed=RatingScaleModel(J,C);fixed.set_parameters(difficulty=np.zeros(J),thresholds=np.arange(C-1,dtype=float))
  pre=r_run(root,'preflight');result['R_execution']='FIXED-PARAMETER PREFLIGHT EXECUTED'
  rp=pd.read_csv(pre/'curves.csv').sort_values(['Node','Item','Category']);P=rp.Probability.to_numpy().reshape(81,J,C);I=rp.Information.to_numpy().reshape(81,J,C)[:,:,0]
  prechecks=dict(probabilities=discrepancies(fixed.probability(theta),P,criteria['mapping_probability']),information=discrepancies(native_information(fixed,theta),I,criteria['mapping_information']))
  result['preflight']=prechecks
  if any(c['status']!='PASS' for c in prechecks.values()):raise ValueError('NOT COMPARABLE: native R/Python RSM probability/information mismatch')
  if prior is not None and (prior/'python_parameters.npz').exists():
   old=json.loads((prior/'report.json').read_text());old_settings=old['settings'];py_meta=old['Python']
   for key in ('data_sha256','stacked_sha256','criteria_sha256','n_quadpts','tol','max_iter','start','bounds','identification'):
    if settings[key]!=old_settings[key]:raise ValueError('Saved Python settings mismatch: '+key)
   if hashlib.sha256((prior/'data.csv').read_bytes()).hexdigest()!=settings['data_sha256'] or not py_meta['converged']:raise ValueError('Saved Python data/convergence mismatch')
   shutil.copytree(prior,root/'original_evidence',dirs_exist_ok=True)
   py=restore_saved_python(dict(np.load(prior/'python_parameters.npz')),py_meta,data,p)
   result['Python_fixed_evaluation']='Public EM: ALL coordinates fixed, one GH evaluation, bit-identical parameters; no parameter optimization'
   result['Python_execution']='RETAINED CONVERGED JOINT FIT FROM RUN 40470531009; NO NEW PYTHON FIT'
  else:
   outcome=engine.fit(data,engine.NAME,p,n_quadpts=61,max_iter=1000,tol=1e-10);py=outcome['fit'];result['Python_execution']='ONE JOINT GLOBAL SHARED-STEP CALIBRATION EXECUTED'
  np.savez(root/'python_parameters.npz',**py.model.parameters)
  py_meta=dict(converged=bool(py.converged),iterations=py.n_iterations,logLik=py.log_likelihood,AIC=py.aic,BIC_original_N=-2*py.log_likelihood+py.n_parameters*np.log(n),Native_stacked_BIC=py.bic,free_parameters=py.n_parameters)
  result['Python']=py_meta;(root/'python_metadata.json').write_text(json.dumps(py_meta,indent=2)+'\n')
  result['R_execution']='CALIBRATION ATTEMPTED';actual=r_run(root,'fit');r=json.loads((actual/'metadata.json').read_text());result['R_execution']='ONE INDEPENDENT JOINT GLOBAL SHARED-STEP CALIBRATION EXECUTED';result['R']=r
  raw=pd.read_csv(actual/'parameters.csv');tau=pd.read_csv(actual/'shared_steps.csv').Tau.to_numpy();native_r=RatingScaleModel(J,C);native_r.set_parameters(difficulty=raw.b.to_numpy(),thresholds=tau)
  rp=pd.read_csv(actual/'curves.csv').sort_values(['Node','Item','Category']);P=rp.Probability.to_numpy().reshape(81,J,C);I=rp.Information.to_numpy().reshape(81,J,C)[:,:,0]
  rgrid=pd.read_csv(actual/'integration_probabilities.csv').sort_values(['Node','Item','Category']).Probability.to_numpy().reshape(61,J,C)
  pll=common_ll(py.model,data,p,grid);rll=sum(likelihood(data.to_numpy()[:,p[:,d].astype(bool)],rgrid[:,p[:,d].astype(bool),:],grid.weights) for d in range(q))
  checks=dict(b=discrepancies(py.model.parameters['difficulty'],raw.b,criteria['calibration_parameters']),shared_steps=discrepancies(py.model.parameters['thresholds'],tau,criteria['calibration_parameters']),
   probabilities=discrepancies(py.model.probability(theta),P,criteria['calibration_probability']),information=discrepancies(native_information(py.model,theta),I,criteria['calibration_information']),
   common_logLik=discrepancies([pll],[rll],criteria['common_logLik']),Python_native_LL=discrepancies([pll],[py.log_likelihood],criteria['native_common_logLik']),R_native_LL=discrepancies([rll],[r['logLik']],criteria['native_common_logLik']))
  for field,tolerance in [('AIC','common_logLik'),('BIC_original_N','common_logLik')]:checks[field]=discrepancies([py_meta[field]],[r[field]],criteria[tolerance])
  rs=pd.read_csv(actual/'scores.csv').sort_values(['Person','Factor']);ps=[];sds=[]
  for d in range(q):
   responses=np.full((16,J),-1,int);responses[:,p[:,d].astype(bool)]=data.to_numpy()[:16,p[:,d].astype(bool)]
   score=mirt.fscores(py,responses,method='EAP',n_quadpts=61);ps.append(np.asarray(score.theta).reshape(-1));sds.append(np.asarray(score.standard_error).reshape(-1))
  checks['EAP']=discrepancies(np.column_stack(ps),rs.EAP.to_numpy().reshape(16,q),criteria['calibration_EAP']);checks['posterior_SD']=discrepancies(np.column_stack(sds),rs.SD.to_numpy().reshape(16,q),criteria['calibration_posterior_SD'])
  # Full information tensors use the same item/factor assignments and zero off-diagonals.
  points=np.column_stack([theta[:,0]]*q);pyc=engine.Curves(py.model,p,C);r_info=np.zeros((81,q,q))
  for d in range(q):r_info[:,d,d]=I[:,p[:,d].astype(bool)].sum(1)
  checks['TIF']=discrepancies(pyc.test_information_matrix(points),r_info,criteria['calibration_information'])
  py_sem=1/np.sqrt(np.diagonal(pyc.test_information_matrix(points),axis1=1,axis2=2));r_sem=1/np.sqrt(np.diagonal(r_info,axis1=1,axis2=2))
  checks['conditional_SEM']=discrepancies(py_sem,r_sem,criteria['calibration_posterior_SD'])
  result['comparison']=checks;result['common_grid_logLik']=dict(Python=pll,R=rll)
  result['gradients']=dict(Python=score_diagnostic(py.model.parameters,data,p,grid),R=score_diagnostic(native_r.parameters,data,p,grid))
  result['boundaries']={name:[dict(parameter=param,coordinates=np.flatnonzero(np.minimum(values+6,6-values)<=np.sqrt(np.finfo(float).eps)*np.maximum(1,abs(values))).tolist()) for param,values in model.parameters.items()] for name,model in [('Python',py.model),('R',native_r)]}
  result['status']='NUMERICALLY VALIDATED / NOT CERTIFIED' if py.converged and r['converged'] and all(c['status']=='PASS' for c in checks.values()) else 'NOT CONVERGED' if not (py.converged and r['converged']) else 'NUMERICAL DISAGREEMENT / NOT CERTIFIED'
 except Exception as exc:result['blocker']=str(exc)
 (root/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result


def execute(out,prior=None):
 out.mkdir(parents=True,exist_ok=True);reports=[]
 for q,n in [(2,200),(3,150)]:
  record=execute_case(q,n,out/f'RSM_{q}D',None if prior is None else prior/f'RSM_{q}D');reports.append(record)
  if record['status']!='NUMERICALLY VALIDATED / NOT CERTIFIED':break
 result=dict(family='RSM',status='NUMERICALLY VALIDATED / NOT CERTIFIED' if len(reports)==2 and all(r['status']=='NUMERICALLY VALIDATED / NOT CERTIFIED' for r in reports) else 'FAIL',profiles=reports,certification='NOT CERTIFIED')
 (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result
