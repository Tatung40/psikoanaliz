"""No reference fabrication or repeated fits: algebra, gates, saved-result policy."""
import numpy as np
import pandas as pd
import pytest
from core import dif_lord_rasch as L
from core.dif_lord_policy import protect_result
from core.dif_lord import link_contrast


def mocked_fit(monkeypatch):
    fit=dict(converged=True,at_bound=False,se_reason='',b=np.array([-.5,.2,.7]),sd=1.4,
        data=np.tile([0.,1.,0.],(50,1)),points=61,loglik=-40.,max_score_per_person=1e-8)
    info=np.eye(4);info[0,3]=info[3,0]=.5
    monkeypatch.setattr(L.rasch,'_information',lambda *a:info)
    monkeypatch.setattr(L.rasch,'_score',lambda *a:np.zeros(4))
    monkeypatch.setattr(L.rasch,'marginal_loglik',lambda *a:-40.)
    return fit,info


def test_full_inverse_nuisance_block_is_not_conditional_information(monkeypatch):
    fit,info=mocked_fit(monkeypatch);b,bb,full,diag=L.rasch_physical_fit(fit)
    np.testing.assert_allclose(full,np.linalg.inv(info));np.testing.assert_allclose(bb,full[:3,:3])
    assert bb[0,0]>np.linalg.inv(info[:3,:3])[0,0]
    assert diag['sd']==1.4 and diag['GH101_score']==0
    # A=1 depends on fixed slopes, not equality of latent SDs.
    d,jr,jf,A,B=link_contrast(b,b+.3,[0,1],2,'1PL')
    assert A==1;np.testing.assert_allclose(d,0,atol=1e-15)


@pytest.mark.parametrize('failure',['nonconverged','bound','singular','illconditioned','audit'])
def test_unchanged_scientific_covariance_and_score_gates(monkeypatch,failure):
    fit,info=mocked_fit(monkeypatch)
    if failure=='nonconverged':fit['converged']=False
    if failure=='bound':fit['at_bound']=True
    if failure=='singular':monkeypatch.setattr(L.rasch,'_information',lambda *a:np.zeros((4,4)))
    if failure=='illconditioned':monkeypatch.setattr(L.rasch,'_information',lambda *a:np.diag([1,1,1,1e-10]))
    if failure=='audit':monkeypatch.setattr(L.rasch,'_score',lambda *a:np.ones(4))
    with pytest.raises(ValueError):L.rasch_physical_fit(fit)


def test_free_scale_inference_requires_actual_independent_reference(monkeypatch):
    result=dict(table=pd.DataFrame([dict(Method='Lord χ²',p=.02,q=.04,p_adj=.04,df=1,Statistic=5.,
        Lord_identification=L.IDENTIFICATION,Status=L.STATUS,Reason='')]),metadata={'Parametric_model':'1PL'})
    monkeypatch.setattr(L,'reference_valid',lambda:False)
    assert protect_result(result)['table'].p.isna().all()
    monkeypatch.setattr(L,'reference_valid',lambda:True)
    assert protect_result(result) is result
    legacy={**result,'table':result['table'].drop(columns='Lord_identification')}
    assert protect_result(legacy)['table'].p.isna().all()


def test_no_reference_cannot_be_assumed_from_model_name(monkeypatch,tmp_path):
    monkeypatch.setattr(L,'REFERENCE',tmp_path/'missing.json')
    assert L.reference_valid() is False
