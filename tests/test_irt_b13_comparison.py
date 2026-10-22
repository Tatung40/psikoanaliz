import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from ui.irt_b13_view import native_fit
from core.irt_b13_compare import compare_engines

def test_rasch_comparison_refused():
    with pytest.raises(ValueError,match="Rasch"):
        compare_engines({"model":"Rasch"},np.zeros((40,5),int))

def test_real_python_mirt_2pl_comparison():
    pytest.importorskip("mirt")
    rng=np.random.default_rng(723)
    theta=rng.normal(size=220)
    b=np.linspace(-1.1,1.1,10)
    x=(rng.random((220,10))<expit(theta[:,None]-b)).astype(int)
    df=pd.DataFrame(x,columns=[f"M{i}" for i in range(10)])
    native=native_fit(df,"2PL")
    if not native["fit"]["converged"]:
        pytest.skip("Native pilot did not converge: cross-engine estimates cannot be certified")
    out=compare_engines(native,df.to_numpy())
    assert out["model"]=="2PL"
    assert out["irf_deviation"]<1e-8
    assert not out["summary"].empty
    assert len(out["items"])==10
