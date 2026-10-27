"""Actual public API blockers and fail-closed application paths; no calibration."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import mirt
from core.irt_multidim import MODELS,fit,support_matrix
from validation.mirt_phase2.capability_audit import TARGETS,audit


@pytest.mark.parametrize('model',TARGETS)
@pytest.mark.parametrize('dimensions',[2,3,4,5])
def test_actual_family_constructor_rejects_multidimensional(model,dimensions):
 kwargs=dict(n_items=10,n_factors=dimensions)
 if model=='RSM':kwargs['n_categories']=4
 with pytest.raises(ValueError,match='unidimensional'):
  TARGETS[model](**kwargs)


@pytest.mark.parametrize('model',TARGETS)
def test_application_never_routes_unsupported_family_to_surrogate(model,monkeypatch):
 def forbidden(*a,**kw):raise AssertionError('Unsupported model reached calibration')
 monkeypatch.setattr(mirt,'fit_mirt',forbidden);monkeypatch.setattr(mirt.EMEstimator,'fit',forbidden)
 data=pd.read_csv('tests/fixtures/irt_b10/data_irt1pl_ok.csv').iloc[:30,:8]
 pattern=np.zeros((8,2),int);pattern[:4,0]=1;pattern[4:,1]=1
 with pytest.raises(ValueError,match='Desteklenmeyen MIRT ailesi'):fit(data,model,pattern)
 assert model not in MODELS
 assert support_matrix().set_index('Model').loc[model,'Status']=='UNSUPPORTED'


def test_no_fit_audit_reports_real_blockers_and_not_scientific_pass(monkeypatch):
 def forbidden(*a,**kw):raise AssertionError('Capability audit attempted calibration')
 monkeypatch.setattr(mirt.EMEstimator,'fit',forbidden)
 result=audit()
 assert not result['calibration_executed'] and result['R_execution']=='NOT EXECUTED'
 assert result['version']=='1.2.0'
 assert len(result['models'])==3 and all(row['status']=='UNSUPPORTED' for row in result['models'])
 assert all(row['scalar_control']['constructed'] and not row['scalar_control']['calibrated'] for row in result['models'])
 assert all(probe['status']=='REJECTED BEFORE FIT' for probe in result['RSM_constraint_probes'])
 assert 'not an equality constraint' in result['RSM_constraint_probes'][0]['reason']
 assert all(len(row['source_sha256'])==64 for row in result['models'])


def test_baseline_workflow_retains_all_accepted_fits_without_precision_refits():
 source=Path('.github/workflows/mirt-stage1.yml').read_text()
 assert 'run-id: 40463908384' in source
 assert '--python-precision' not in source and '--refine-failed' not in source
 assert '--reuse original-evidence/mirt-stage1 --expand' in source
 assert 'tests/test_mirt_stage1.py tests/test_mirt_multidim_ui.py' in source
