import numpy as np
import pandas as pd
from core.irt import fit_irt
from core.irt_hessian import observed_information_se


def _data():
    rng=np.random.default_rng(817)
    ability=rng.normal(size=500)
    b=np.linspace(-1,1,5)
    prob=1/(1+np.exp(-(ability[:,None]-b[None,:])))
    return pd.DataFrame((rng.random(prob.shape)<prob).astype(int),
                        columns=[f"M{i+1}" for i in range(5)])


def test_observed_hessian_se_has_safe_output():
    data=_data()
    fit=fit_irt(data,model="1PL",points=21,max_iter=20)
    tab,reason=observed_information_se(data,fit["items"],"1PL",points=21)
    if reason:
        assert tab is None
    else:
        assert len(tab)==5
        assert (tab["SH(b)"]>0).all()
        assert np.isfinite(tab["SH(b)"]).all()


def test_observed_hessian_refuses_boundaries():
    data=_data()
    fit=fit_irt(data,model="3PL",points=15,max_iter=5)
    table=fit["items"].copy()
    table.loc[0,"a"]=4.
    output,reason=observed_information_se(data,table,"3PL",points=15)
    assert output is None
    assert "sınır" in reason
