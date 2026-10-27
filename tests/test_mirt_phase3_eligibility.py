"""Production scope must be backed by ACTUAL converged independent profiles."""
import hashlib,json
from pathlib import Path
import numpy as np
from core import irt_multidim_general_report as report


def test_registry_eligibility_requires_real_convergence_and_exact_scope():
 root=Path('validation/mirt_phase3');cases=[]
 initial=json.loads((root/'fixed_3PL_initial_evidence.json').read_text());cases.extend(initial['cases'])
 for name in ('expansion_evidence.json','free4_precision_evidence.json'):
  evidence=json.loads((root/name).read_text())
  cases.extend(c for stage in evidence['stages'] for c in stage['cases'])
 criteria=hashlib.sha256(Path('validation/mirt_stage1/criteria.json').read_bytes()).hexdigest()
 accepted=[]
 for c in cases:
  if c['status']!='NUMERICALLY VALIDATED / NOT CERTIFIED':continue
  assert c['Python']['converged'] and c['R']['converged'] and c['R']['constraints_valid']
  assert c['settings']['criteria_sha256']==criteria
  assert all(v['status']=='PASS' for v in c['comparison'].values())
  cfg=c['settings'];within=bool((np.array(cfg['pattern']).sum(1)>1).any())
  accepted.append((cfg['family'],cfg['q'],within,cfg['free_c'],cfg['free_u']))
 for label,(family,free_c,free_u) in report.CONFIG.items():
  for q in report.allowed_dimensions(label):
   for within in ([False] if free_c else [False,True]):
    assert (family,q,within,free_c,free_u) in accepted,(label,q,within)
 original=json.loads((root/'free4_original_evidence.json').read_text())['stages'][0]['cases'][0]
 assert original['status']=='NOT CONVERGED' and not original['R']['converged']
 trace=json.loads((root/'free4_native_convergence.json').read_text())
 assert trace['reported_converged'] and trace['exact_last_pair_satisfied'] and trace['max_change']<1e-8
 # Missing R convergence on the genuine3D resource block NEVER becomes support.
 assert ('3PL',3,True,True,False) not in accepted and ('4PL',3,True,True,True) not in accepted


def test_runtime_contract_is_explicitly_not_irt_validation():
 source=Path('validation/mirt_phase3/runtime_contract.R').read_text()
 assert 'IRT_fits_executed=FALSE' in source
 assert "optimizer(.1,function(p)sum((p-.2)^2),method='nlminb'" in source
 assert 'mirt(' not in source and 'mirt::mirt' not in source
 assert 'reloadPars C++ mutates S4 objects' in source
