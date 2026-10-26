"""No calibration: verified public API shapes, mapping, identification, budgets."""
import json,inspect
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import mirt
from scipy.special import expit
from core.irt_multidim import structure,fit,support_matrix
from validation.mirt_stage1.compare import profile,inputs,PROFILES,masses,likelihood

@pytest.mark.parametrize('name',PROFILES)
def test_fixed_parameter_mapping_and_information(name,tmp_path):
 cfg,data,engine,grid=inputs(name,tmp_path/name);q=cfg['q'];pattern=np.array(cfg['pattern'])
 a=pattern*1.2;theta=np.array([[-.5]*q,[0]*q,[.5]*q]);p=engine.probability(theta)
 if cfg['model']=='2PL':
  np.testing.assert_allclose(p,expit(theta@a.T+np.linspace(-.7,.7,cfg['k'])),atol=1e-14)
  p=np.stack([1-p,p],-1)
 else:
  b=np.tile(np.linspace(-1,1,3),(cfg['k'],1));cum=expit((theta@a.T)[:,:,None]-a.sum(1)[None,:,None]*b)
  expected=np.concatenate([np.ones((len(theta),cfg['k'],1)),cum,np.zeros((len(theta),cfg['k'],1))],2)
  np.testing.assert_allclose(p,-np.diff(expected,axis=2),atol=1e-14)
 grad=[];h=1e-5
 for j in range(q):
  delta=np.zeros(q);delta[j]=h
  hi=engine.probability(theta+delta);lo=engine.probability(theta-delta)
  if hi.ndim==2:hi=np.stack([1-hi,hi],-1);lo=np.stack([1-lo,lo],-1)
  grad.append((hi-lo)/(2*h))
 grad=np.stack(grad,-1)
 manual=np.einsum('njcr,njcs,njc->njrs',grad,grad,1/p)
 actual=np.stack([engine.item_information_matrix(theta,j) for j in range(cfg['k'])],1)
 np.testing.assert_allclose(actual,manual,rtol=1e-6,atol=1e-8)
 np.testing.assert_allclose(engine.test_information_matrix(theta),actual.sum(1),atol=1e-12)
 assert all(set(data[c])==set(range(2 if cfg['model']=='2PL' else 4)) for c in data)
 weight=masses(grid.nodes,grid.weights,np.eye(q));np.testing.assert_allclose(weight,grid.weights,atol=1e-16)
 assert np.isfinite(likelihood(data.to_numpy(),engine.probability(grid.nodes),weight))
 # Positive pure anchors fix factor order/sign; no unverified free rotation.
 assert np.linalg.matrix_rank(pattern)==q
 assert all(np.sum(pattern[:,j]&(pattern.sum(1)==1))>=3 for j in range(q))


def test_reject_unidentified_and_unsupported_models():
 with pytest.raises(ValueError,match='saf madde'):structure(np.tile([[1,1,0],[0,1,1],[1,0,1]],(3,1)))
 with pytest.raises(ValueError):structure(np.ones((18,6)))
 with pytest.raises(ValueError):fit(pd.DataFrame(np.ones((50,6))),'3PL',np.ones((6,2)))
 matrix=support_matrix().set_index('Model')
 assert matrix.loc['2PL','Phase1'] and matrix.loc['GRM','Phase1']
 assert not matrix.loc['GPCM','Phase1'] and not matrix.loc['NRM','Phase1']
 assert not matrix.loc['3PL','Multidimensional_API'] and not matrix.loc['4PL','Multidimensional_API']


def test_public_fit_contract_and_resource_gate(monkeypatch):
 assert 'n_factors' in inspect.signature(mirt.fit_mirt).parameters
 cfg=profile('2PL_2D');d=pd.DataFrame(np.tile([0,1],(60,5)),columns=[f'M{i}' for i in range(10)])
 # Ensure every column has both codes without running any calibration.
 d.iloc[::2,:]=0;d.iloc[1::2,:]=1
 calls=[]
 def sentinel(*a,**kw):calls.append(kw);raise RuntimeError('API contract reached')
 monkeypatch.setattr(mirt,'fit_mirt',sentinel)
 with pytest.raises(RuntimeError,match='API contract'):fit(d,'2PL',cfg['pattern'])
 assert len(calls)==1 and calls[0]['n_factors']==2 and calls[0]['compute_standard_errors'] is False
 assert calls[0]['priors'] is None and calls[0]['accelerate']=='none'
 with pytest.raises(ValueError,match='bellek'):fit(d,'2PL',cfg['pattern'],n_quadpts=100)


def test_installed_bounds_and_preregistered_criteria():
 from mirt.estimation.base import _ITEM_PARAMETER_BOUNDS
 from mirt.estimation._graded_order import THRESHOLD_GAP
 assert _ITEM_PARAMETER_BOUNDS['discrimination']==(.1,5.)
 assert _ITEM_PARAMETER_BOUNDS['thresholds']==(-6.,6.)
 assert _ITEM_PARAMETER_BOUNDS['intercepts']==(-6.,6.) and THRESHOLD_GAP==1e-6
 criteria=json.loads(Path('validation/mirt_stage1/criteria.json').read_text())
 assert criteria['scope'] and criteria['calibration_probability']==.0001
 assert all(criteria[k]>0 for k in criteria if k.startswith(('mapping_','calibration_')))


def test_r_parameter_table_preserves_native_attributes():
 source=Path('validation/mirt_stage1/reference.R').read_text()
 assert 'native_attributes<-attributes(sv)' in source
 assert 'same_attributes(sv,native_attributes)' in source
 assert 'identical(observed[[n]],expected[[n]])' in source
 assert 'native_column_classes' in source
 assert "!same_attributes(bad,native_attributes)" in source
 assert "inherits(sv,'mirtValues')" not in source
 assert "sv<-do.call(mirt,c(base,list(pars='values')))" in source


def test_r_metadata_precision_and_gaussian_ratio():
 source=Path('validation/mirt_stage1/reference.R').read_text()
 assert 'digits=NA' in source and 'mvtnorm::' not in source
 assert 'source_free_parameters' in source and "mode!='fit')sv$est[]<-FALSE" in source
 from scipy.stats import multivariate_normal
 points=np.array([[-2.,1.],[0.,0.],[.2,-.6]])
 for rho in (-.4,0.,.25):
  x,y=points.T
  ratio=-.5*np.log1p(-rho*rho)+(2*rho*x*y-rho*rho*(x*x+y*y))/(2*(1-rho*rho))
  expected=multivariate_normal.logpdf(points,cov=[[1,rho],[rho,1]])-multivariate_normal.logpdf(points,cov=np.eye(2))
  np.testing.assert_allclose(ratio,expected,atol=1e-14)


def test_recovery_keeps_original_failure_and_never_refits(tmp_path,monkeypatch):
 """Harness isolation only: mocked reevaluation is not scientific R evidence."""
 import hashlib
 import validation.mirt_stage1.compare as harness
 source=tmp_path/'prior'/'GRM_2D';source.mkdir(parents=True)
 content=b'A,B\n0,1\n';(source/'data.csv').write_bytes(content)
 original=dict(profile='GRM_2D',settings=dict(criteria_sha256=hashlib.sha256((harness.ROOT/'criteria.json').read_bytes()).hexdigest(),data_sha256=hashlib.sha256(content).hexdigest()),
  Python=dict(converged=True),R=dict(converged=True),comparison=dict(other=dict(status='PASS'),R_native_LL=dict(status='FAIL')),
  common_grid_logLik=dict(R=-123.1234567890123))
 (source/'report.json').write_text(json.dumps(original))
 def nofit(*a,**k):raise AssertionError('Unexpected recalibration')
 def fixed(root,mode):
  assert mode=='evaluate_saved';target=root/mode;target.mkdir()
  (target/'metadata.json').write_text(json.dumps(dict(logLik=original['common_grid_logLik']['R'],constraints_valid=True)))
  return target
 monkeypatch.setattr(harness,'fit',nofit);monkeypatch.setattr(harness,'r_run',fixed)
 result=harness.recover('GRM_2D',tmp_path/'prior',tmp_path/'recovered')
 assert result['status']=='NUMERICALLY VALIDATED / NOT CERTIFIED'
 assert json.loads((tmp_path/'recovered'/'original_report.json').read_text())['comparison']['R_native_LL']['status']=='FAIL'


@pytest.mark.parametrize('name',__import__('validation.mirt_stage1.compare',fromlist=['EXPANSION']).EXPANSION)
def test_expansion_fixed_parameter_probabilities_information(name,tmp_path):
 from scipy.special import logsumexp
 cfg,data,engine,grid=inputs(name,tmp_path/name);q=cfg['q']
 theta=np.array([[-.5]*q,[0.]*q,[.5]*q]);p=engine.probability(theta)
 if p.ndim==2:p=np.stack((1-p,p),-1)
 assert np.isfinite(p).all();np.testing.assert_allclose(p.sum(-1),1,atol=1e-14)
 if cfg['model']=='NRM':logits=np.einsum('nq,jcq->njc',theta,engine.parameters['slopes'])+engine.parameters['intercepts']
 elif cfg['model'] in ('PCM','GPCM'):
  a=engine.parameters['discrimination'];steps=engine.parameters['steps']
  logits=(theta@a.T)[:,:,None]*np.arange(4)-a.sum(1)[None,:,None]*np.column_stack((np.zeros(cfg['k']),np.cumsum(steps,axis=1)))
 else:logits=None
 if logits is not None:np.testing.assert_allclose(p,np.exp(logits-logsumexp(logits,axis=2,keepdims=True)),atol=1e-14)
 h=1e-5;der=[]
 for j in range(q):
  delta=np.zeros(q);delta[j]=h;hi=engine.probability(theta+delta);lo=engine.probability(theta-delta)
  if hi.ndim==2:hi=np.stack((1-hi,hi),-1);lo=np.stack((1-lo,lo),-1)
  der.append((hi-lo)/(2*h))
 g=np.stack(der,-1);manual=np.einsum('njcr,njcs,njc->njrs',g,g,1/p)
 actual=np.stack([engine.item_information_matrix(theta,j) for j in range(cfg['k'])],1)
 np.testing.assert_allclose(actual,manual,rtol=1e-6,atol=1e-8)
 assert all(set(data[c])==set(range(2 if cfg['model'] in ('2PL','Rasch','Bifactor') else 4)) for c in data)


def test_canonical_nominal_gauge_and_exploratory_guard(tmp_path):
 from types import SimpleNamespace
 from core.irt_multidim import _canonical
 cfg,_,engine,_=inputs('NRM_2D',tmp_path/'nominal');pattern=np.array(cfg['pattern'],bool)
 original=engine.parameters['slopes'].copy();engine.set_parameters(slopes=original*np.array([-1,1]))
 native=SimpleNamespace(model=engine,latent_covariance=np.array([[1,.2],[.2,1]]))
 changed,cov,signs=_canonical(native,native.latent_covariance,pattern,'NRM')
 np.testing.assert_array_equal(signs,[-1,1]);np.testing.assert_array_equal(changed.model.parameters['slopes'],original)
 assert cov[0,1]==-.2 and engine.parameters['slopes'][0,1,0]<0
 pat=np.ones((8,2),int);pat[:2]=np.tril(np.ones((2,2),int))
 assert structure(pat,exploratory=True)
 with pytest.raises(ValueError):structure(pat,correlated=True,exploratory=True)
