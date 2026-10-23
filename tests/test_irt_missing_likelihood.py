import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit
from core.irt_missing_likelihood import marginal_loglik_missing

def test_no_missing_matches_direct_quadrature():
    x=np.array([[1.,0.,1.],[0.,1.,0.],[1.,1.,0.]])
    a=np.array([1.,1.4,.8]);b=np.array([0.,-.5,.3])
    nodes,w=hermgauss(21)
    theta=np.sqrt(2)*nodes
    p=expit(a[None,:]*(theta[:,None]-b[None,:]))
    direct=0.
    for row in x:
        terms=np.prod(p**row[None,:]*(1-p)**(1-row)[None,:],axis=1)
        direct+=np.log(np.dot(w/np.sqrt(np.pi),terms))
    assert abs(marginal_loglik_missing(x,a,b,points=21)-direct)<1e-10

def test_missing_is_marginalized_not_zero_scored():
    x=np.array([[1.,np.nan,0.],[np.nan,np.nan,np.nan]])
    a=np.array([1.,1.,1.]);b=np.array([0.,0.,0.])
    missing_ll=marginal_loglik_missing(x,a,b)
    zero_ll=marginal_loglik_missing(np.nan_to_num(x),a,b)
    assert missing_ll>zero_ll
    assert np.isfinite(missing_ll)

def test_invalid_response_rejected():
    import pytest
    with pytest.raises(ValueError):
        marginal_loglik_missing(np.array([[1.,2.,0.]]),np.ones(3),np.zeros(3))
