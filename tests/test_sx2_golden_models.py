"""S-X² golden regressions from independently supplied sx2_ref.py.

Uses deterministic synthetic 200x8 data; no private Ornek.xlsx data is
committed. Reference calculated with fixed item parameters, not re-fitted.
"""
import numpy as np
import pandas as pd
import pytest
from core.irt_itemfit import sx2_mirt_compatible

REFERENCE = {
 "1PL": (
 [12.499874533,9.9535533838,6.2112705466,8.7065489892,6.2124207907,1.0449649723,3.3593837821,15.2139940672],
 [5,6,6,6,6,5,5,5],
 [.0285445468,.1266216637,.3999445482,.1907665178,.3998203183,.9588581173,.6447630412,.0094860585]),
 "2PL": (
 [6.3966686355,9.3908375126,4.838295008,8.5828111658,6.8664235594,1.2963339093,1.5812556452,15.9984433406],
 [5,5,5,5,4,4,4,4],
 [.2695113023,.0944542642,.4359321256,.1269068311,.1431177656,.8619972308,.8121570669,.0030212532]),
 "3PL": (
 [6.1846779213,8.2349185355,6.4638602355,8.230979055,6.1290799185,1.9049642339,3.8102778124,16.2136289123],
 [4,4,4,4,4,3,3,3],
 [.1857743932,.0833421222,.1670817716,.0834743042,.1897136605,.5923640536,.28269291,.0010251653]),
 "4PL": (
 [11.6911437591,10.1032114908,10.1088426642,7.1541120338,5.2468777044,5.6840006052,8.6535864949,17.9078997109],
 [3,3,3,3,3,3,2,2],
 [.0085196535,.0177088696,.0176632341,.0671445879,.1545867961,.128038298,.0132098403,.0001292257])
}

@pytest.mark.parametrize("model", ["1PL","2PL","3PL","4PL"])
def test_fixed_parameter_goldens(model):
    rng=np.random.default_rng(311)
    n,k=200,8
    theta=rng.normal(size=n)
    a=np.linspace(.7,1.6,k)
    b=np.linspace(-1,1,k)
    p=1/(1+np.exp(-(theta[:,None]*a-a*b)))
    data=pd.DataFrame((rng.random((n,k))<p).astype(int),
                      columns=[f"M{i+1}" for i in range(k)])
    aa=np.ones(k) if model=="1PL" else a
    cc=np.linspace(.01,.12,k) if model in ("3PL","4PL") else np.zeros(k)
    dd=np.linspace(.82,.98,k) if model=="4PL" else np.ones(k)
    pars=pd.DataFrame({"Madde":data.columns,"a":aa,"b":b,"c":cc,"d":dd})
    actual=sx2_mirt_compatible(data,pars,model)
    expected_stat,expected_df,expected_p=REFERENCE[model]
    np.testing.assert_allclose(actual["S-X²"],expected_stat,rtol=1e-9,atol=1e-8)
    np.testing.assert_array_equal(actual["sd"],expected_df)
    np.testing.assert_allclose(actual["p"],expected_p,rtol=1e-8,atol=1e-8)
    assert actual["Durum"].eq("Hesaplandı").all()
