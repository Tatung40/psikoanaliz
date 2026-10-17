import numpy as np
import pandas as pd
import pytest
from core.irt_multistart import fit_irt_multistart

def _responses():
    rng=np.random.default_rng(24)
    theta=rng.normal(size=120)
    a=np.array([.8,1.,1.4,1.2])
    b=np.array([-.9,-.2,.3,.9])
    prob=.1+.9/(1+np.exp(-a[None,:]*(theta[:,None]-b[None,:])))
    return pd.DataFrame((rng.random(prob.shape)<prob).astype(int),
                        columns=["M1","M2","M3","M4"])

@pytest.mark.parametrize("model",["3PL","4PL"])
def test_multistart_returns_audited_candidate_and_keeps_failures_visible(model):
    result=fit_irt_multistart(_responses(),model=model,
                             starts=(0.,-1.,1.),points=11,max_iter=3)
    assert len(result["candidates"])==3
    assert result["fit"]["model"]==model
    assert result["summary"]["Başarılı kestirim"]>=1
    assert len(result["boundary"])==result["summary"]["Sınırdaki parametre"]
    assert np.isfinite(result["summary"]["Seçilen logLik"])
    assert result["fit"]["loglik"]==result["summary"]["Seçilen logLik"]

def test_multistart_refuses_invalid_inputs():
    x=_responses()
    with pytest.raises(ValueError,match="3PL/4PL"):
        fit_irt_multistart(x,model="2PL")
    with pytest.raises(ValueError,match="başlangıç"):
        fit_irt_multistart(x,starts=())
