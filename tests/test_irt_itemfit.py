"""Reference fixed-parameter S-X² regression checks (mirt-style oracle)."""
import numpy as np
import pandas as pd
from core.irt_itemfit import sx2_mirt_compatible, _collapse_cells


def test_sx2_matches_supplied_reference_oracle():
    rng=np.random.default_rng(311)
    n,k=200,8
    theta=rng.normal(size=n)
    a=np.linspace(.7,1.6,k)
    b=np.linspace(-1,1,k)
    p=1/(1+np.exp(-(theta[:,None]*a-a*b)))
    x=pd.DataFrame((rng.random((n,k))<p).astype(int),
                   columns=[f"M{i+1}" for i in range(k)])
    pars=pd.DataFrame({"Madde":x.columns,"a":a,"b":b,
                       "c":np.zeros(k),"d":np.ones(k)})
    out=sx2_mirt_compatible(x,pars,"2PL")
    expected=np.array([6.3966686355,9.39083751258,4.83829500801,
                       8.58281116583,6.86642355942,1.2963339093,
                       1.58125564524,15.9984433406])
    df=np.array([5,5,5,5,4,4,4,4])
    np.testing.assert_allclose(out["S-X²"],expected,atol=1e-8,rtol=1e-9)
    np.testing.assert_array_equal(out["sd"],df)
    assert out["p"].between(0,1).all()


def test_collapse_preserves_cell_totals():
    obs=np.array([[1.,0.],[0.,3.],[2.,1.],[1.,2.]])
    exp=np.array([[.4,.6],[1.4,1.6],[.7,2.3],[2.,1.]])
    a,b,reason=_collapse_cells(obs,exp)
    assert not reason
    np.testing.assert_allclose(a.sum(0),obs.sum(0))
    np.testing.assert_allclose(b.sum(0),exp.sum(0))
    assert not np.any((b>0)&(b<1))
