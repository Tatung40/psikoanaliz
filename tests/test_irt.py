"""MTK regresyon ve sayısal kararlılık testleri."""
import numpy as np
import pandas as pd
import pytest
from core.irt import probabilities, information, fit_irt, validate_binary, estimate_ability, approximate_q3, approximate_parameter_uncertainty

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


def test_approximate_q3_matrix_properties():
    data=make_data(n=200,k=5,seed=18)
    fit=fit_irt(data,model="2PL",points=15,max_iter=8)
    q3=approximate_q3(data,fit["items"],fit["theta"])
    assert q3.shape==(5,5)
    assert list(q3.columns)==list(data.columns)
    assert np.isnan(np.diag(q3)).all()
    assert np.allclose(q3.to_numpy(),q3.to_numpy().T,equal_nan=True)
    assert np.isfinite(q3.to_numpy()[~np.eye(5,dtype=bool)]).all()


def test_irt_fit_loglik_consistency():
    from scipy.special import logsumexp
    from numpy.polynomial.hermite import hermgauss
    data=make_data(n=140,k=4,seed=23)
    fitted=fit_irt(data,model="2PL",points=15,max_iter=10)
    a,b,c,d=(fitted["items"][key].to_numpy()
             for key in ("a","b","c","d"))
    gh,w=hermgauss(15)
    p=np.clip(probabilities(np.sqrt(2)*gh,a,b,c,d),1e-9,1-1e-9)
    x=data.to_numpy()
    direct=logsumexp(x@np.log(p).T+(1-x)@np.log1p(-p).T
                     +np.log(w/np.sqrt(np.pi))[None,:],axis=1).sum()
    assert np.isclose(direct,fitted["loglik"],rtol=1e-9,atol=1e-7)


@pytest.mark.parametrize("model",["1PL","2PL","3PL","4PL"])
def test_opg_uncertainty_schema_and_safety(model):
    data=make_data(n=180,k=4,seed=91)
    fitted=fit_irt(data,model=model,points=15,max_iter=8)
    table, reason=approximate_parameter_uncertainty(
        data,fitted["items"],model,points=15)
    assert len(table)==4
    assert all(f"SE({p})" in table.columns for p in ("a","b","c","d"))
    if reason:
        assert table[[f"SE({p})" for p in ("a","b","c","d")]].isna().all().all()
    else:
        estimated={"1PL":{"b"},"2PL":{"a","b"},
                   "3PL":{"a","b","c"},"4PL":{"a","b","c","d"}}[model]
        for p in estimated:
            assert np.isfinite(table[f"SE({p})"]).all()
            assert (table[f"SE({p})"]>0).all()
