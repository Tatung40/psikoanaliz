import numpy as np
import pytest
from core.irt_missing_em import fit_irt_missing
from core.irt_missing_likelihood import marginal_loglik_missing

@pytest.mark.parametrize("model",["1PL","2PL","3PL","4PL"])
def test_missing_em_fits_and_likelihood_recomputes(model):
    rng=np.random.default_rng(20261023)
    n,k=90,8
    theta=rng.standard_normal(n)
    b=np.linspace(-1.2,1.2,k)
    a=np.linspace(.8,1.5,k)
    p=1/(1+np.exp(-a[None,:]*(theta[:,None]-b[None,:])))
    x=(rng.random((n,k))<p).astype(float)
    mask=rng.random((n,k))<.08
    x[mask]=np.nan
    result=fit_irt_missing(x,model=model,points=11,max_iter=3)
    assert result["missing_count"]==int(mask.sum())
    assert result["items"].shape==(k,4)
    assert np.isfinite(result["loglik"])
    t=result["items"]
    independently=marginal_loglik_missing(x,t["a"].to_numpy(),t["b"].to_numpy(),
                                         t["c"].to_numpy(),t["d"].to_numpy(),points=11)
    assert abs(independently-result["loglik"])<1e-7
    assert result["status"]=="experimental_requires_three_way_validation"

def test_reject_nonbinary_observations():
    x=np.ones((30,4))
    x[0,0]=2
    with pytest.raises(ValueError):
        fit_irt_missing(x)
