import numpy as np
import pandas as pd
import pytest
from core.irt_multistart import fit_irt_multistart

def _responses(k=4):
    rng=np.random.default_rng(24)
    theta=rng.normal(size=120)
    a=np.array([.8,1.,1.4,1.2,1.1])[:k]
    b=np.array([-.9,-.2,.3,.9,0.])[:k]
    prob=.1+.9/(1+np.exp(-a[None,:]*(theta[:,None]-b[None,:])))
    return pd.DataFrame((rng.random(prob.shape)<prob).astype(int),
                        columns=[f"M{i+1}" for i in range(k)])

@pytest.mark.parametrize("model",["3PL","4PL"])
def test_multistart_returns_audited_candidate_and_keeps_failures_visible(model):
    data=_responses(5 if model=="4PL" else 4)   # 4PL needs k>=5 (2**k-1 >= 4k)
    result=fit_irt_multistart(data,model=model,
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


def test_multistart_reports_structural_unidentifiability_once():
    with pytest.raises(ValueError,match="yapısal olarak tanımlanamaz"):
        fit_irt_multistart(_responses(4),model="4PL",starts=(0.,-1.,1.))
