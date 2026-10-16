"""Independent derivative and information-matrix regression tests.

R mirt benchmark results (Ornek.xlsx, N=200, k=25) are recorded as
external validation evidence, not asserted as an exact-fit golden test:
different calibrations produce different SE estimates. No private data
are included in the repository.
"""
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit, logsumexp
from scipy.optimize import minimize
from core.irt_hessian import observed_information_se


def _reference_loglik(data, a, b, points=21):
    gh,w=hermgauss(points)
    theta=np.sqrt(2.)*gh
    p=expit(theta[:,None]*a[None,:]-a[None,:]*b[None,:])
    x=data.to_numpy()
    return float(logsumexp(
        x@np.log(p).T+(1-x)@np.log1p(-p).T
        +np.log(w/np.sqrt(np.pi))[None,:],axis=1).sum())


def _fixture(n=400,k=4):
    rng=np.random.default_rng(729)
    z=rng.standard_normal(n)
    a=np.array([.7,1.0,1.2,1.4])[:k]
    b=np.array([-.8,-.3,.4,.9])[:k]
    p=expit(z[:,None]*a[None,:]-a[None,:]*b[None,:])
    x=pd.DataFrame((rng.random((n,k))<p).astype(int),
                   columns=[f"M{i+1}" for i in range(k)])
    params=pd.DataFrame({"Madde":x.columns,"a":a,"b":b,
                         "c":np.zeros(k),"d":np.ones(k)})
    return x,params


def _reference_se(data,pars,model,points):
    a=pars["a"].to_numpy(float); b=pars["b"].to_numpy(float)
    if model=="1PL":
        v=b.copy()
        def unpack(z):return np.ones(len(b)),z
    else:
        v=np.column_stack([a,b]).reshape(-1)
        def unpack(z):
            t=z.reshape(-1,2)
            return t[:,0],t[:,1]
    def objective(z):
        aa,bb=unpack(z)
        return -_reference_loglik(data,aa,bb,points)
    steps=1e-4*np.maximum(1,np.abs(v))
    h=np.zeros((len(v),len(v)))
    f0=objective(v)
    for i in range(len(v)):
        ei=np.zeros(len(v));ei[i]=steps[i]
        h[i,i]=(objective(v+ei)-2*f0+objective(v-ei))/steps[i]**2
        for j in range(i):
            ej=np.zeros(len(v));ej[j]=steps[j]
            h[i,j]=(objective(v+ei+ej)-objective(v+ei-ej)
                    -objective(v-ei+ej)+objective(v-ei-ej))/(4*steps[i]*steps[j])
            h[j,i]=h[i,j]
    vals=np.linalg.eigvalsh(h)
    if vals[0]<=1e-6:return None
    return np.sqrt(np.diag(np.linalg.inv(h)))


def test_one_pl_information_matches_direct_loglik_hessian():
    x,p=_fixture()
    p["a"]=1.
    z=minimize(lambda b:-_reference_loglik(x,np.ones(len(b)),b,21),
               p["b"].to_numpy(),method="L-BFGS-B",
               bounds=[(-3.5,3.5)]*len(p))
    assert z.success
    p["b"]=z.x
    observed,reason=observed_information_se(x,p,"1PL",points=21)
    assert not reason, reason
    independent=_reference_se(x,p,"1PL",21)
    assert independent is not None
    np.testing.assert_allclose(observed["SH(b)"],independent,rtol=.003,atol=.0006)


def test_two_pl_information_matches_direct_loglik_hessian():
    x,p=_fixture()
    initial=np.column_stack([p["a"],p["b"]]).reshape(-1)
    def objective(z):
        t=z.reshape(-1,2)
        return -_reference_loglik(x,t[:,0],t[:,1],21)
    z=minimize(objective,initial,method="L-BFGS-B",
               bounds=[(.2,3.7),(-3.5,3.5)]*len(p))
    assert z.success
    t=z.x.reshape(-1,2)
    p["a"]=t[:,0]
    p["b"]=t[:,1]
    observed,reason=observed_information_se(x,p,"2PL",points=21)
    assert not reason, reason
    independent=_reference_se(x,p,"2PL",21)
    assert independent is not None
    np.testing.assert_allclose(observed["SH(a)"],independent[::2],rtol=.008,atol=.0015)
    np.testing.assert_allclose(observed["SH(b)"],independent[1::2],rtol=.008,atol=.0015)


def test_near_bound_is_not_misreported_as_valid_se():
    x,p=_fixture()
    p.loc[0,"a"]=3.99999
    values,reason=observed_information_se(x,p,"2PL",points=21)
    assert values is None
    assert "sınır" in reason
