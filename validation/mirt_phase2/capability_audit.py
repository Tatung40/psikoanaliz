"""No-fit public API audit: constructor acceptance never certifies a model."""
import argparse,hashlib,inspect,json,platform,shutil
from pathlib import Path
import mirt
from mirt.models.base import BaseItemModel
from mirt.models.dichotomous import ThreeParameterLogistic,FourParameterLogistic
from mirt.models.polytomous import RatingScaleModel
from mirt.estimation.constraints import LinearConstraint,ConstraintSet

TARGETS={'3PL':ThreeParameterLogistic,'4PL':FourParameterLogistic,'RSM':RatingScaleModel}


def audit():
 result=dict(scope='Native full multidimensional constructors only; excludes exact independent 1D block composition',engine='Python mirt',version=mirt.__version__,python=platform.python_version(),
  calibration_executed=False,R_execution='NOT EXECUTED',
  R_runtime='AVAILABLE / NO FIT REQUESTED' if shutil.which('Rscript') else 'R VALIDATION BLOCKED: Rscript unavailable locally',
  R_comparison='NOT COMPARABLE for native full multidimensional constructors; restricted composition is validated separately',
  accepted_baseline=dict(run=40453192278,profiles=12,status='Retained PR80 evidence; no recalibration'),models=[])
 for name,cls in TARGETS.items():
  kwargs=dict(n_items=10)
  if name=='RSM':kwargs['n_categories']=4
  control=cls(n_factors=1,**kwargs)
  row=dict(model=name,class_name=cls.__module__+'.'+cls.__name__,
   constructor_signature=str(inspect.signature(cls)),supports_multidimensional=cls.supports_multidimensional,
   source_sha256=hashlib.sha256(inspect.getsource(cls).encode()).hexdigest(),
   scalar_control=dict(dimensions=control.n_factors,constructed=True,calibrated=False),conditions=[])
  for q in (2,3,4,5):
   entry=dict(dimensions=q,structures=['between-item','within-item'],calibration_executed=False)
   try:
    cls(n_factors=q,**kwargs)
    entry.update(status='EXPERIMENTAL / NOT CERTIFIED',reason='Constructor accepted; independent equivalence and inference still unverified')
   except Exception as exc:
    text=str(exc);expected=not cls.supports_multidimensional and 'unidimensional' in text.lower()
    entry.update(status='UNSUPPORTED' if expected else 'BLOCKED',exception=type(exc).__name__,reason=text)
   row['conditions'].append(entry)
  row['status']='UNSUPPORTED' if all(c['status']=='UNSUPPORTED' for c in row['conditions']) else 'NOT CERTIFIED'
  result['models'].append(row)
 # LinearConstraint exists as a projection utility, but the actual ML fitter
 # rejects it. Projecting fitted GPCM steps afterward is not constrained RSM ML.
 probes=[]
 linear=LinearConstraint('steps',item_indices=[0,1],coefficients=[1.,-1.],target=0.,factor=0)
 for name,constraints in [('LinearConstraint',[linear]),('ConstraintSet',ConstraintSet([linear]))]:
  try:
   mirt.EMEstimator(constraints=constraints,compute_standard_errors=False)
   probes.append(dict(api=name,status='API CHANGED / INDEPENDENT VALIDATION REQUIRED'))
  except Exception as exc:probes.append(dict(api=name,status='REJECTED BEFORE FIT',exception=type(exc).__name__,reason=str(exc)))
 result['RSM_constraint_probes']=probes
 result['base_constructor_guard_sha256']=hashlib.sha256(inspect.getsource(BaseItemModel.__init__).encode()).hexdigest()
 return result


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 result=audit();args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 print(json.dumps({r['model']:r['status'] for r in result['models']}))
 # Success verifies safe closure, never a scientific PASS or support claim.
 raise SystemExit(0 if all(r['status']=='UNSUPPORTED' for r in result['models']) and all(r['status']=='REJECTED BEFORE FIT' for r in result['RSM_constraint_probes']) else 1)
