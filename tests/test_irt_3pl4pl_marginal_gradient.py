"""Independent finite-difference validation of 3PL/4PL marginal score."""
import numpy as np
import pandas as pd
import pytest
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit, logsumexp
from core.irt_hessian import marginal_score


def _fixture(model):
    rng=np.random.default_rng(491)
    k=4
    a=np.array([.8,1.,1.2,1.5])
    b=np.array([-.7,-.2,.3,.8])
    c=np.array([.08,.1,.12,.14])
    d=np.ones(k) if model=="3PL" else np.array([.91,.93,.95,.94])
    z=rng.standard_normal(600)
    p=c[None,:]+(d-c)[None,:]*expit(a[None,:]*(z[:,None]-b[None,:]))
    x=pd.DataFrame((rng.random((len(z),k))<p).astype(int),
                   columns=[f"M{i+1}" for i in range(k)])
    items=pd.DataFrame({"Madde":x.columns,"a":a,"b":b,"c":c,"d":d})
    return x,items


def _independent_ll(x,items,points=21):
    gh,w=hermgauss(points)
    theta=np.sqrt(2)*gh
    a,b,c,d=[items[q].to_numpy(float) for q in ("a","b","c","d")]
    p=c[None,:]+(d-c)[None,:]*expit(a[None,:]*(theta[:,None]-b[None,:]))
    y=x.to_numpy()
    return float(logsumexp(y@np.log(p).T+(1-y)@np.log1p(-p).T
                           +np.log(w/np.sqrt(np.pi))[None,:],axis=1).sum())


@pytest.mark.parametrize("model",["3PL","4PL"])
def test_analytic_marginal_score_matches_independent_finite_difference(model):
    x,items=_fixture(model)
    params=("a","b","c") if model=="3PL" else ("a","b","c","d")
    actual=marginal_score(x,items,model,points=21)
    numeric=[]
    for j in range(len(items)):
        for q in params:
            step=1e-5
            up=items.copy()
            down=items.copy()
            up.at[j,q]+=step
            down.at[j,q]-=step
            numeric.append((_independent_ll(x,up)-_independent_ll(x,down))/
                           (2*step))
    np.testing.assert_allclose(actual,numeric,rtol=1e-5,atol=3e-5)
