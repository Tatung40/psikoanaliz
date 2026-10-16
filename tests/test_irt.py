"""MTK regresyon ve sayısal kararlılık testleri."""
import numpy as np
import pandas as pd
import pytest
from core.irt import probabilities, information, fit_irt, validate_binary, estimate_ability

def make_data(n=180,k=5,seed=77):
    rng=np.random.default_rng(seed)
    theta=rng.normal(size=n)
    a=np.linspace(.7,1.4,k)
    b=np.linspace(-1.,1.,k)
    p=probabilities(theta,a,b,np.zeros(k),np.ones(k))
    return pd.DataFrame((rng.random(p.shape)<p).astype(int),
                        columns=[f"M{i+1}" for i in range(k)])

def test_probability_bounds_and_information():
    t=np.linspace(-4,4,100)
    a=np.array([1.,1.5]);b=np.array([0.,1.])
    c=np.array([0.,.1]);d=np.array([1.,.9])
    p=probabilities(t,a,b,c,d)
    assert p.shape==(100,2)
    assert np.all((p>0)&(p<1))
    assert np.all(information(t,a,b,c,d)>=0)
    assert p[0,1]<p[-1,1]

@pytest.mark.parametrize("model",["1PL","2PL","3PL","4PL"])
def test_mml_models(model):
    x=make_data()
    r=fit_irt(x,model=model,points=15,max_iter=5)
    assert np.isfinite(r["loglik"])
    assert len(r["items"])==x.shape[1]
    assert r["icc"].shape==(161,x.shape[1])
    assert r["iif"].shape==(161,x.shape[1])
    assert np.isfinite(r["theta"]).all()
    assert np.isfinite(r["tif"]).all()
    assert np.all(r["items"]["d"] > r["items"]["c"])
    if model=="1PL":
        assert np.all(r["items"]["a"]==1.)

def test_invalid_input():
    with pytest.raises(ValueError):
        validate_binary(pd.DataFrame(np.ones((100,4))))
    with pytest.raises(ValueError):
        fit_irt(np.zeros((10,2)))

@pytest.mark.parametrize("method",["EAP","MAP","ML"])
def test_ability_methods(method):
    data=make_data()
    fitted=fit_irt(data,model="2PL",points=15,max_iter=3)
    scored=estimate_ability(data,fitted["items"],method=method,points=61)
    assert len(scored)==len(data)
    assert np.isfinite(scored["Yetenek (theta)"]).all()
    assert np.isfinite(scored["Standart Hata"]).all()
    assert (scored["Standart Hata"]>=0).all()
