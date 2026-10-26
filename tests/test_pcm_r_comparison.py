"""PCM identities and validation configuration; no calibration or simulation."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import mirt
from validation import grm_r_comparison as G


def test_pcm_likelihood_unordered_steps_and_scores():
    b=np.array([[.5,-.5,.2],[-1.,.3,1.2]])
    a=np.ones(2);theta=np.array([-2.,-1.,0.,1.,2.]);w=np.array([.05,.2,.5,.2,.05])
    x=np.array([[0,1],[2,3],[1,2],[3,0]])
    ll,g=G.marginal(x,a,b,theta,w,model='PCM')
    model=mirt.PartialCreditModel(2,4);model.set_parameters(steps=b)
    P=model.probability(theta)
    direct=sum(np.log(sum(w[q]*np.prod([P[q,i,row[i]] for i in range(2)]) for q in range(5))) for row in x)
    np.testing.assert_allclose(ll,direct,rtol=1e-13)
    for ix in np.ndindex(b.shape):
        up,dn=b.copy(),b.copy();up[ix]+=1e-5;dn[ix]-=1e-5
        diff=(G.marginal(x,a,up,theta,w,'PCM')[0]-G.marginal(x,a,dn,theta,w,'PCM')[0])/2e-5
        np.testing.assert_allclose(g['b'][ix],diff,rtol=1e-6,atol=1e-8)
    with pytest.raises(ValueError,match='fixed a=1'):G.marginal(x,a+.01,b,theta,w,'PCM')


def test_pcm_fit_is_single_ml_call_and_retains_unordered_starts(monkeypatch):
    calls=[];b=np.array([[.5,-.5,.2]]*3)
    def fake(x,**kw):
        calls.append(kw)
        return SimpleNamespace(model=SimpleNamespace(steps=b,discrimination=np.ones(3)),latent_mean=None,latent_covariance=None,
            converged=True,n_iterations=2,log_likelihood=-10.,aic=38.,bic=41.)
    monkeypatch.setattr(mirt,'fit_mirt',fake)
    seed=pd.DataFrame(dict(item=list('ABC'),a=[1.]*3,b1=b[:,0],b2=b[:,1],b3=b[:,2]))
    _,p,meta=G.python_once(pd.DataFrame([[0,1,2]],columns=list('ABC')),seed,'PCM')
    assert len(calls)==1 and calls[0]['model']=='PCM' and calls[0]['n_quadpts']==61
    assert set(calls[0]['start_values'])=={'steps'} and calls[0]['priors'] is None
    np.testing.assert_array_equal(calls[0]['start_values']['steps'],b)
    np.testing.assert_array_equal(p[['b1','b2','b3']],b)
    assert meta['converged'] and meta['variance']==1


def test_pcm_extreme_probabilities_remain_finite():
    ll,g=G.marginal(np.array([[0],[1],[2],[3]]),np.ones(1),np.array([[6.,-6.,6.]]),np.array([-14.,14.]),np.array([.5,.5]),'PCM')
    assert np.isfinite(ll) and np.isfinite(g['b']).all()
