import numpy as np
import pandas as pd
from validation.r_mirt_phase1.equivalence_audit import evaluate


def test_common_grid_likelihood_enumerated_binary_patterns():
    p=pd.DataFrame(dict(a=[1.], b=[0.], c=[0.], d=[1.]))
    prob, info, ll=evaluate(np.array([[0],[1]]),p,np.array([-1.,1.]),np.array([.5,.5]))
    assert np.allclose(prob.sum(axis=0),1)
    assert np.allclose(info,prob[:,0]*(1-prob[:,0]))
    assert np.isclose(ll,2*np.log(.5))


def test_asymptote_derivative_information():
    p=pd.DataFrame(dict(a=[2.], b=[0.], c=[.2], d=[.9]))
    prob,info,_=evaluate(np.array([[1]]),p,np.array([0.]),np.array([1.]))
    assert np.isclose(prob[0,0],.55)
    assert np.isclose(info[0],(.7*2*.25)**2/(.55*.45))


def test_extreme_slopes_and_exact_asymptotes_stay_finite():
    p=pd.DataFrame(dict(a=[17.465], b=[0.], c=[0.], d=[1.]))
    prob,info,ll=evaluate(np.array([[0],[1]]),p,np.array([-14.,14.]),np.array([.5,.5]))
    assert np.isfinite(info).all() and np.isfinite(ll)
    assert np.isclose(ll,2*np.log(.5))
