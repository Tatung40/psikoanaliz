"""One public Python/R ML calibration per unique independent item block."""
import argparse,hashlib,json,platform,shutil,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import mirt
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.models.dichotomous import ThreeParameterLogistic,FourParameterLogistic
from core import irt_multidim_factorized as engine
from validation.r_mirt_phase1.harness import discrepancies
from validation.mirt_stage1.compare import likelihood

ROOT=Path(__file__).resolve().parent
CRITERIA=ROOT.parent/'mirt_stage1/criteria.json'


def dataset(family):
 source=Path(f'tests/fixtures/irt_b11/data_irt{family.lower()}_ok.csv');raw=pd.read_csv(source)
 rng=np.random.default_rng(20261027);n=600;blocks=[raw.iloc[:n,:6],raw.iloc[:n,6:12],raw.iloc[n:2*n,:6]]
 parts=[];maps=[]
 for d,block in enumerate(blocks):
  order=np.arange(n) if d==0 else rng.permutation(n)
  frame=block.iloc[order].reset_index(drop=True).copy();frame.columns=[f'F{d+1}_{name}' for name in frame]
  parts.append(frame);maps.append(order.tolist())
 data=pd.concat(parts,axis=1).astype(int);pattern=np.zeros((18,3),int)
 for d in range(3):pattern[d*6:(d+1)*6,d]=1
 return data,pattern,dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),seed=20261027,
  kind='Synthetic composite respondents from disjoint existing item blocks; not real respondent IDs',row_maps=maps)


def r_run(family,block,mode):
 out=block/mode;out.mkdir(exist_ok=True)
 with (out/'stdout.log').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
  run=subprocess.run(['Rscript',str(ROOT/'reference.R'),family,str(block.resolve()),mode],stdout=stdout,stderr=stderr,timeout=1200)
 if run.returncode:raise RuntimeError(f'R {mode} failed ({run.returncode}); original stderr retained')
 return out


def compare_block(py,block,data,grid,criteria,d):
 actual=block/'fit';theta=np.linspace(-4,4,81)[:,None]
 r=json.loads((actual/'metadata.json').read_text());raw=pd.read_csv(actual/'parameters.csv')
 params=py.model.parameters;pp=py.model.probability(theta);rp=pd.read_csv(actual/'curves.csv').sort_values(['Node','Item']);rgrid=pd.read_csv(actual/'integration_probabilities.csv').to_numpy()
 pLL=likelihood(data.to_numpy(),py.model.probability(grid.nodes),grid.weights);rLL=likelihood(data.to_numpy(),rgrid,grid.weights)
 score=mirt.fscores(py,data.iloc[:16].to_numpy(),method='EAP',n_quadpts=61);rs=pd.read_csv(actual/'scores.csv')
 checks=dict(a=discrepancies(params['discrimination'],raw.a,criteria['calibration_parameters']),b=discrepancies(params['difficulty'],raw.b,criteria['calibration_parameters']),
  probabilities=discrepancies(pp,rp.Probability.to_numpy().reshape(81,6),criteria['calibration_probability']),
  information=discrepancies(py.model.information(theta),rp.Information.to_numpy().reshape(81,6),criteria['calibration_information']),
  EAP=discrepancies(np.asarray(score.theta).reshape(-1),rs.EAP,criteria['calibration_EAP']),SD=discrepancies(np.asarray(score.standard_error).reshape(-1),rs.SD,criteria['calibration_posterior_SD']),
  common_logLik=discrepancies([pLL],[rLL],criteria['common_logLik']),Python_native_LL=discrepancies([pLL],[py.log_likelihood],criteria['native_common_logLik']),R_native_LL=discrepancies([rLL],[r['logLik']],criteria['native_common_logLik']))
 status='NUMERICALLY VALIDATED / NOT CERTIFIED' if py.converged and r['converged'] and all(c['status']=='PASS' for c in checks.values()) else 'NOT CONVERGED' if not (py.converged and r['converged']) else 'NUMERICAL DISAGREEMENT / NOT CERTIFIED'
 record=dict(factor=d+1,status=status,comparison=checks,Python=dict(converged=bool(py.converged),iterations=py.n_iterations,logLik=py.log_likelihood,AIC=py.aic,BIC=py.bic,free_parameters=py.n_parameters),R=r)
 (block/'report.json').write_text(json.dumps(record,indent=2)+'\n');return record


def execute(family,out):
 out.mkdir(parents=True,exist_ok=True);data,pattern,provenance=dataset(family)
 criteria=json.loads(CRITERIA.read_text());model=next((v for v in engine.MODELS if v.startswith(family)),None)
 settings=dict(family=family,mean=0,variance=1,c=.2,u=.95 if family=='4PL' else 1.,D=1,priors=None,
  n_quadpts=61,max_iter=1000,tol=1e-10,criteria_sha256=hashlib.sha256(CRITERIA.read_bytes()).hexdigest(),
  Python='mirt '+mirt.__version__,python=platform.python_version(),dataset=provenance,
  parameter_bounds=dict(a=[.1,5],b=[-6,6]),identification='Named orthogonal positive-slope axes; no rotations or free covariance',
  starting_values='a=1,b=0,c=.2,u=.95(4PL)/1(3PL); all asymptotes fixed',
  R_stopping='Maximum parameter change <1e-8, maximum 3000 EM cycles; SLSQP; no acceleration')
 data.to_csv(out/'data.csv',index=False);settings['data_sha256']=hashlib.sha256((out/'data.csv').read_bytes()).hexdigest()
 (out/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
 result=dict(status='BLOCKED',settings=settings,blocks=[],R_execution='NOT EXECUTED',Python_execution='NOT EXECUTED',certification='NOT CERTIFIED')
 try:
  if model is None:raise ValueError('Family not yet enabled after preceding validation stage')
  if not shutil.which('Rscript'):raise RuntimeError('R VALIDATION BLOCKED: Rscript unavailable')
  grid=GaussHermiteQuadrature(61,1);theta=np.linspace(-4,4,81)[:,None]
  for d in range(3):
   block=out/f'F{d+1}';block.mkdir(exist_ok=True)
   data.iloc[:,d*6:(d+1)*6].to_csv(block/'data.csv',index=False)
   pd.DataFrame(dict(theta=grid.nodes[:,0],weight=grid.weights)).to_csv(block/'integration.csv',index=False,float_format='%.17g')
  pre=r_run(family,out/'F1','preflight');result['R_execution']='FIXED-PARAMETER PREFLIGHT EXECUTED'
  cls=ThreeParameterLogistic if family=='3PL' else FourParameterLogistic
  fixed=cls(6);values=dict(discrimination=np.ones(6),difficulty=np.zeros(6),guessing=np.full(6,.2))
  if family=='4PL':values['upper']=np.full(6,.95)
  fixed.set_parameters(**values);rp=pd.read_csv(pre/'curves.csv').sort_values(['Node','Item'])
  prechecks=dict(probabilities=discrepancies(fixed.probability(theta),rp.Probability.to_numpy().reshape(81,6),criteria['mapping_probability']),
   information=discrepancies(fixed.information(theta),rp.Information.to_numpy().reshape(81,6),criteria['mapping_information']))
  result['preflight']=prechecks
  if any(c['status']!='PASS' for c in prechecks.values()):raise ValueError('NOT COMPARABLE: native R/Python fixed model mismatch')
  fitted=engine.fit(data,model,pattern,c=.2,u=settings['u'],n_quadpts=61,max_iter=1000,tol=1e-10);result['Python_execution']='THREE UNIQUE BLOCK CALIBRATIONS EXECUTED'
  result['Python_blocks']=[]
  for d,index,py in fitted['blocks']:
   block=out/f'F{d+1}';np.savez(block/'python_parameters.npz',**py.model.parameters)
   record=dict(factor=d+1,converged=bool(py.converged),iterations=py.n_iterations,logLik=py.log_likelihood,AIC=py.aic,BIC=py.bic,free_parameters=py.n_parameters)
   (block/'python_metadata.json').write_text(json.dumps(record,indent=2)+'\n');result['Python_blocks'].append(record)
  for d,index,py in fitted['blocks']:
   block=out/f'F{d+1}';result['R_execution']=f'CALIBRATION ATTEMPTED F{d+1}'
   actual=r_run(family,block,'fit');result['R_completed_blocks']=d+1;record=compare_block(py,block,data.iloc[:,index],grid,criteria,d)
   (block/'report.json').write_text(json.dumps(record,indent=2)+'\n');result['blocks'].append(record)
  result['R_execution']='THREE UNIQUE INDEPENDENT BLOCK CALIBRATIONS EXECUTED'
  result['assemblies']=[dict(dimensions=q,status='NUMERICALLY VALIDATED / NOT CERTIFIED' if all(b['status']=='NUMERICALLY VALIDATED / NOT CERTIFIED' for b in result['blocks'][:q]) else 'FAIL',
   Python_logLik=sum(b['Python']['logLik'] for b in result['blocks'][:q]),R_logLik=sum(b['R']['logLik'] for b in result['blocks'][:q]),
   method='Exact product likelihood; no additional calibration') for q in (2,3)]
  result['status']='NUMERICALLY VALIDATED / NOT CERTIFIED' if all(a['status']=='NUMERICALLY VALIDATED / NOT CERTIFIED' for a in result['assemblies']) else 'FAIL'
 except Exception as exc:result['blocker']=str(exc)
 (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result


def continue_four(prior,out):
 """One justified Python F2 continuation; all R fits and accepted blocks retained."""
 old=json.loads((prior/'report.json').read_text());cfg=old['settings'];criteria=json.loads(CRITERIA.read_text())
 if cfg['family']!='4PL' or cfg['criteria_sha256']!=hashlib.sha256(CRITERIA.read_bytes()).hexdigest() or cfg['data_sha256']!=hashlib.sha256((prior/'data.csv').read_bytes()).hexdigest():raise ValueError('Continuation data/family/criteria mismatch')
 if not all(b['R']['converged'] for b in old['blocks']) or any(old['blocks'][i]['status']!='NUMERICALLY VALIDATED / NOT CERTIFIED' for i in (0,2)):raise ValueError('Reference/accepted blocks changed')
 shutil.copytree(prior,out,dirs_exist_ok=True);shutil.copy2(out/'report.json',out/'original_report.json')
 result=old;result['status']='BLOCKED';result['R_execution']='RETAINED THREE CONVERGED INDEPENDENT R FITS / NO NEW R FIT';result['Python_execution']='F2 CONTINUATION ATTEMPTED; F1/F3 RETAINED'
 result['settings']['continuation']=dict(factor=2,source_run=40463908384,max_iter=3000,tol=1e-12,item_optim_ftol=1e-14,item_optim_maxiter=200,initialization='Own saved Python F2 fit; no R parameters copied')
 try:
  data=pd.read_csv(out/'data.csv');block=out/'F2';part=pd.read_csv(block/'data.csv');pd.testing.assert_frame_equal(part,data.iloc[:,6:12])
  start=dict(np.load(block/'python_parameters.npz'));shutil.copy2(block/'python_parameters.npz',block/'previous_python_parameters.npz');shutil.copy2(block/'report.json',block/'previous_report.json')
  native=FourParameterLogistic(6,item_names=list(part.columns));native.set_parameters(**start);native.set_free_parameter_masks(dict(guessing=np.zeros(6,bool),upper=np.zeros(6,bool)))
  estimator=mirt.EMEstimator(n_quadpts=61,max_iter=3000,tol=1e-12,item_optim_ftol=1e-14,item_optim_maxiter=200,compute_standard_errors=False,accelerate='none')
  py=estimator.fit(native,part.to_numpy(dtype=int),start=start)
  if not np.array_equal(py.model.parameters['guessing'],np.full(6,.2)) or not np.array_equal(py.model.parameters['upper'],np.full(6,.95)):raise ValueError('Fixed asymptotes changed')
  np.savez(block/'python_parameters.npz',**py.model.parameters)
  record=compare_block(py,block,part,GaussHermiteQuadrature(61,1),criteria,1);record['previous_Python']=old['blocks'][1]['Python'];record['Python']['stopping']='Public EM absolute LL change <1e-12; item ftol1e-14; max3000'
  (block/'report.json').write_text(json.dumps(record,indent=2)+'\n');result['blocks'][1]=record
  result['Python_execution']='ONLY F2 PYTHON CONTINUATION EXECUTED; F1/F3 RETAINED'
  result['assemblies']=[dict(dimensions=q,status='NUMERICALLY VALIDATED / NOT CERTIFIED' if all(b['status']=='NUMERICALLY VALIDATED / NOT CERTIFIED' for b in result['blocks'][:q]) else 'FAIL',Python_logLik=sum(b['Python']['logLik'] for b in result['blocks'][:q]),R_logLik=sum(b['R']['logLik'] for b in result['blocks'][:q]),method='Exact product likelihood; reference and accepted blocks retained') for q in (2,3)]
  result['status']='NUMERICALLY VALIDATED / NOT CERTIFIED' if all(a['status']=='NUMERICALLY VALIDATED / NOT CERTIFIED' for a in result['assemblies']) else 'FAIL'
 except Exception as exc:result['blocker']=str(exc)
 (out/'settings.json').write_text(json.dumps(result['settings'],indent=2)+'\n');(out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--family',choices=['none','3PL','4PL','4PL_CONTINUE'],default=(ROOT/'requested_family.txt').read_text().strip());parser.add_argument('--prior',type=Path,default=Path('original-evidence/mirt-phase2-fit'));parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 if args.family=='none':print('Phase 2 calibration NOT REQUESTED; no model validation claim');raise SystemExit(0)
 result=continue_four(args.prior,args.output) if args.family=='4PL_CONTINUE' else execute(args.family,args.output);print('PHASE2_REPORT='+json.dumps(result));raise SystemExit(0 if result['status']=='NUMERICALLY VALIDATED / NOT CERTIFIED' else 1)
