"""Independent math, no fitted-data certificates."""
import numpy as np
from scipy import linalg
from scipy.special import expit
from validation.binary_global_fit import kernel as K

def eta(model,J=12):
    a=np.linspace(.8,2,J);b=np.linspace(-1.25,1.25,J)
    if model=="Rasch":return b
    if model=="common1PL":return np.r_[b,1.2]
    return np.stack([a,b,np.full(J,.15),np.full(J,.85)][:{"2PL":2,"3PL":3,"4PL":4}[model]],axis=1).ravel()

def test_union_covariance_matches_full_independent_pattern_enumeration():
    J=8;e=eta("4PL",J);t,w=K.grid(61);a,b,c,d=K.unpack(e,"4PL",J)
    P=c+(d-c)*expit(a*(t[:,None]-b));X=((np.arange(2**J)[:,None]>>np.arange(J))&1)
    px=np.prod(np.where(X[:,None,:],P[None],1-P[None]),axis=2)@w
    Z=np.stack([np.prod(X[:,S],axis=1) for S in K.structure(J)[0]],axis=1)
    mu,Xi,D=K.moments(e,"4PL",J)
    assert np.max(abs(mu-px@Z))<1e-12
    assert np.max(abs(Xi-((px[:,None]*Z).T@Z-np.outer(mu,mu))))<1e-10

def test_full_jacobians_at_two_independent_steps():
    for model in ("Rasch","common1PL","2PL","3PL","4PL"):
        e=eta(model,7);mu,Xi,D=K.moments(e,model,7)
        for h in (1e-5,5e-6):
            numerical=np.column_stack([(K.moments(e+v,model,7,jacobian=False)[0]-K.moments(e-v,model,7,jacobian=False)[0])/(2*h) for v in np.eye(len(e))*h])
            assert np.max(abs(D-numerical))<1e-7+1e-5*np.max(abs(D))

def test_projection_matches_independent_euclidean_nullspace():
    J=7;e=eta("2PL",J);mu,Xi,D=K.moments(e,"2PL",J)
    observed=mu+np.random.default_rng(45).normal(0,.003,len(mu))
    stat,diag,_,_=K.m2_from_moments(observed,mu,Xi,D,1000)
    Q=linalg.null_space(D.T,rcond=1e-10)
    v=Q.T@(observed-mu)
    ref=float(1000*v@np.linalg.solve(Q.T@Xi@Q,v))
    assert abs(stat-ref)<1e-7
    assert diag["df"]==len(mu)-D.shape[1] and diag["rank_stable"]

def test_independence_baseline_and_boundary_refusal():
    J=6;p=np.linspace(.25,.75,J);mu,Xi=K.moment_core(p[None],np.ones(1))
    stat,diag=K.baseline(mu,J,1000)
    assert stat<1e-20 and diag["df"]==J*(J-1)//2
    import pytest
    bad=mu.copy();bad[0]=0.
    with pytest.raises(ValueError,match="boundary"):K.baseline(bad,J,1000)
    with pytest.raises(ValueError,match="binary"):K.observed(np.array([[0,np.nan,1],[1,0,1]]))
