"""Small deterministic checks, with NO model calibration."""
import numpy as np
import pandas as pd
import mirt
import pytest
from validation import grm_r_comparison as G


def test_common_grid_loglik_and_scores_against_independent_category_formula():
    a=np.array([1.2,.9]);b=np.array([[-1.,0.,1.],[-.5,.3,1.2]])
    theta=np.array([-2.,-1.,0.,1.,2.]);weights=np.array([.05,.2,.5,.2,.05])
    data=np.array([[0,1],[2,3],[1,2],[3,0]])
    ll,g=G.marginal(data,a,b,theta,weights)
    model=mirt.GradedResponseModel(2,4);model.set_parameters(discrimination=a,thresholds=b)
    P=model.probability(theta)
    direct=sum(np.log(sum(weights[q]*np.prod([P[q,i,row[i]] for i in range(2)]) for q in range(5))) for row in data)
    np.testing.assert_allclose(ll,direct,rtol=1e-13)
    for name,values in [('a',a),('b',b)]:
        for index in np.ndindex(values.shape):
            ap,am,bp,bm=a.copy(),a.copy(),b.copy(),b.copy()
            plus,minus=(ap,am) if name=='a' else (bp,bm)
            plus[index]+=1e-5;minus[index]-=1e-5
            numerical=(G.marginal(data,ap,bp,theta,weights)[0]-G.marginal(data,am,bm,theta,weights)[0])/2e-5
            np.testing.assert_allclose(g[name][index],numerical,rtol=1e-6,atol=1e-8)


def test_tail_likelihood_does_not_clip_or_lose_rare_categories():
    theta=np.array([-14.,14.]);weights=np.array([.5,.5])
    ll,score=G.marginal(np.array([[0],[1],[2],[3]]),np.array([5.]),np.array([[-1.,0.,1.]]),theta,weights)
    assert np.isfinite(ll) and np.isfinite(score['a']).all() and np.isfinite(score['b']).all()


def test_public_python_route_is_single_grm_ml_call(monkeypatch):
    from types import SimpleNamespace
    calls=[]
    def fake(data,**kw):
        calls.append(kw)
        return SimpleNamespace(model=SimpleNamespace(discrimination=np.ones(2),thresholds=np.array([[-1.,0.,1.]]*2)),
            latent_mean=np.array([0.]),latent_covariance=np.array([[1.]]),converged=False,n_iterations=3,log_likelihood=-8.,aic=32.,bic=33.)
    monkeypatch.setattr(mirt,'fit_mirt',fake)
    seed=pd.DataFrame(dict(item=['A','B'],a=[1.,1.],b1=[-1.,-1.],b2=[0.,0.],b3=[1.,1.]))
    _,_,meta=G.python_once(pd.DataFrame([[0,1],[1,2]],columns=['A','B']),seed)
    assert len(calls)==1 and calls[0]['model']=='GRM' and calls[0]['n_quadpts']==61
    assert calls[0]['priors'] is None and not calls[0]['compute_standard_errors']
    assert meta['converged'] is False


def test_unexecuted_preflight_stops_before_any_calibration(tmp_path,monkeypatch):
    import json
    pre=tmp_path/'pre';pre.mkdir();(pre/'report.json').write_text(json.dumps(dict(status='NOT EXECUTED')))
    monkeypatch.setattr(mirt,'fit_mirt',lambda *a,**kw:pytest.fail('fit forbidden'))
    r=G.execute(pre,tmp_path/'out',allow_fit=True,rscript='mock-R')
    assert r['status']=='FAIL' and r['fit_calls']==dict(R=0,python=0) and not r['scientific_pass']
