"""Derivative and identification tests; NOT independent calibration evidence."""
import numpy as np
import pandas as pd
import pytest
from scipy.optimize._numdiff import approx_derivative
from validation.mirt_phase3 import engine


@pytest.mark.parametrize('q',[2,3])
@pytest.mark.parametrize('family,free_c,free_u',[('2PL',False,False),('3PL',False,False),('3PL',True,False),('4PL',False,False),('4PL',True,True)])
def test_exact_physical_scores_covariance_and_information(q,family,free_c,free_u):
 pattern=np.zeros((3*q+2,q),int)
 for d in range(q):pattern[3*d:3*d+3,d]=1
 pattern[-2:]=1
 spec=engine.Specification(pattern,family,True,free_c,free_u,c=.2,u=.95 if family=='4PL' else 1.)
 x,_=spec.start_bounds();x[-spec.n_cov:]=np.linspace(-.2,.3,spec.n_cov)
 data=pd.read_csv('tests/fixtures/irt_b11/data_irt3pl_ok.csv').iloc[:30,:len(pattern)].to_numpy()
 obj=engine.Objective(data,spec,5);value,score=obj.evaluate(x)
 for step in (1e-4,5e-5):
  numerical=approx_derivative(lambda v:obj.evaluate(v)[0],x,method='3-point',abs_step=step).reshape(-1)
  np.testing.assert_allclose(score,numerical,atol=2e-6,rtol=2e-5)
 assert np.linalg.eigvalsh(spec.covariance(x[-spec.n_cov:])).min()>0
 np.testing.assert_allclose(np.diag(spec.covariance(x[-spec.n_cov:])),1,atol=1e-15)
 theta=np.array([[0.]*q,[.3]*q]);p=engine.probabilities(spec,x,theta);grad=[]
 for d in range(q):
  delta=np.zeros(q);delta[d]=1e-5;grad.append((engine.probabilities(spec,x,theta+delta)-engine.probabilities(spec,x,theta-delta))/(2e-5))
 grad=np.stack(grad,-1);expected=np.einsum('njr,njs,nj->njrs',grad,grad,1/(p*(1-p)))
 np.testing.assert_allclose(engine.information(spec,x,theta),expected,atol=1e-9,rtol=1e-7)
 mean,cov=engine.eap(obj,x);assert mean.shape==(30,q) and cov.shape==(30,q,q)
 assert np.linalg.eigvalsh(cov).min()>0


def test_unidentified_high_dimensional_and_invalid_data_are_rejected():
 with pytest.raises(ValueError):engine.Specification(np.ones((12,2)))
 with pytest.raises(ValueError):engine.Specification(np.eye(4))
 p=np.repeat(np.eye(2),3,axis=0);spec=engine.Specification(p)
 with pytest.raises(ValueError):engine.Objective(np.ones((30,6)),spec)
 with pytest.raises(ValueError):engine.Specification(p,'4PL',c=.5,u=.5)
 with pytest.raises(ValueError):engine.Specification(p,'3PL',free_u=True)


def test_no_r_blocks_before_any_fit_and_preserves_inputs(tmp_path,monkeypatch):
 from validation.mirt_phase3 import compare
 monkeypatch.setattr(compare.shutil,'which',lambda _:None)
 def forbidden(*a,**kw):raise AssertionError('Fit before independent reference preflight')
 monkeypatch.setattr(engine,'fit',forbidden)
 result=compare.run_case('3PL',2,False,False,tmp_path/'run')
 assert result['status']=='BLOCKED' and result['Python_execution']=='NOT EXECUTED'
 assert 'Rscript unavailable' in result['blocker'] and (tmp_path/'run/settings.json').exists()


def test_saved_stage_a_is_real_correlated_2pl_not_new_calibration():
 import json,hashlib
 from pathlib import Path
 root=Path('validation/mirt_phase3');record=json.loads((root/'stage_A_evidence.json').read_text())
 assert record['status']=='NUMERICALLY VALIDATED / NOT CERTIFIED' and record['Python']['converged'] and record['R']['converged']
 assert record['data_sha256']==hashlib.sha256((root/'fixtures/2PL_2D.csv').read_bytes()).hexdigest()
 assert all(v['status']=='PASS' for v in record['comparison'].values())
 assert not np.allclose(record['R']['covariance'],np.eye(2))


def test_saturated_logistic_tails_use_exact_log_probabilities_without_clipping():
 pattern=np.zeros((11,3),int)
 for d in range(3):pattern[3*d:3*d+3,d]=1
 pattern[-2:]=1
 spec=engine.Specification(pattern,'3PL');x,_=spec.start_bounds();x[:len(spec.active)]=5
 theta=np.array([[1000.]*3,[-1000.]*3]);lp,lq,_,_=engine.log_probabilities(spec,x,theta)
 assert np.isfinite(lp).all() and np.isfinite(lq).all()
 assert np.isfinite(engine.information(spec,x,theta)).all()
 data=pd.read_csv('tests/fixtures/irt_b11/data_irt3pl_ok.csv').iloc[:30,:11].to_numpy()
 value,score=engine.Objective(data,spec,15).evaluate(x)
 assert np.isfinite(value) and np.isfinite(score).all()


def test_q_matrix_values_cannot_be_silently_coerced_to_boolean():
 pattern=np.repeat(np.eye(2),3,axis=0);pattern[-1,0]=.3
 with pytest.raises(ValueError,match='0/1'):engine.Specification(pattern)


def test_unit_variance_representation_guard_is_machine_precision_only():
 from pathlib import Path
 source=Path('validation/mirt_phase3/reference.R').read_text()
 assert 'all(diag(S)==1)' not in source and '8*.Machine$double.eps' in source
 assert '!unit_diagonal(1+1e-12)' in source
 assert source.index("'native_fit_status.json'")<source.index('unit_diagonal(diag(S))')
