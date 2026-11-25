"""Exact marginal-score differentiation and sampling covariance algebra; no fits."""
import numpy as np
import pytest
from validation.parameter_uncertainty_g.runner import marginal_scores,empirical_sandwich
from validation.binary_certification.common import Objective


@pytest.mark.parametrize("model",["3PL","4PL"])
def test_marginal_score_full_physical_gradient(model):
    x=np.array([[0,1,0],[1,0,1],[1,1,1],[0,0,0]],float)
    p={"a":np.array([.8,1.2,1.5]),"b":np.array([-1.,0.,1.]),
       "c":np.array([.1,.15,.2]),"d":np.full(3,1. if model=="3PL" else .93)}
    scores=marginal_scores(x,p)
    objective=Objective(x,model,map_priors=False,points=61)
    for q in ("a","b","c")+(("d",) if model=="4PL" else ()):
        for j in range(3):
            for h in (1e-5,2e-5):
                low={n:v.copy() for n,v in p.items()};high={n:v.copy() for n,v in p.items()}
                low[q][j]-=h;high[q][j]+=h
                fd=(objective.value(high)-objective.value(low))/(2*h)
                assert scores[q][:,j].sum()==pytest.approx(fd,abs=2e-7)


def test_joint_empirical_meat_differs_from_failed_a1_and_includes_cross_covariance():
    V=np.array([[.5,.1],[.1,.8]])
    scores=np.array([[1.,2.],[-1.,0.],[2.,-1.],[0.,1.]])
    S,B=empirical_sandwich(V,scores)
    np.testing.assert_allclose(B,len(scores)*np.cov(scores,rowvar=False),atol=1e-14)
    np.testing.assert_allclose(S,V@B@V.T,atol=1e-14)
    assert B[0,1]!=0 and S[0,1]!=0
    shifted=empirical_sandwich(V,scores+[20.,-10.])[0]
    np.testing.assert_allclose(S,shifted,atol=1e-14)


def test_singular_sampling_covariance_rejected():
    with pytest.raises(ValueError):empirical_sandwich(np.eye(2),np.ones((5,2)))
