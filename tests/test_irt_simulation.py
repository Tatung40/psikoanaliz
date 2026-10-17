import numpy as np
import pandas as pd
import pytest
from core.irt_simulation import simulate_irt, parameter_recovery


def _items(model):
    k=5
    return pd.DataFrame({
        "Madde":[f"M{j+1}" for j in range(k)],
        "a":np.ones(k) if model=="1PL" else np.array([.8,1.1,1.4,1.,1.2]),
        "b":np.array([-.8,-.4,0.,.5,.9]),
        "c":np.full(k,.12 if model in ("3PL","4PL") else 0.),
        "d":np.full(k,.92 if model=="4PL" else 1.)
    })


@pytest.mark.parametrize("model",["1PL","2PL","3PL","4PL"])
def test_simulation_deterministic_and_shape(model):
    params=_items(model)
    x,t=simulate_irt(120,params,seed=327)
    x2,t2=simulate_irt(120,params,seed=327)
    pd.testing.assert_frame_equal(x,x2)
    np.testing.assert_array_equal(t,t2)
    assert x.shape==(120,5)
    assert set(np.unique(x.to_numpy())).issubset({0,1})


@pytest.mark.parametrize("model",["3PL","4PL"])
def test_truth_probabilities_are_recovered_by_simulated_frequencies(model):
    params=_items(model)
    x,t=simulate_irt(10000,params,seed=12)
    from core.irt import probabilities
    a,b,c,d=[params[q].to_numpy(float) for q in ("a","b","c","d")]
    predicted=probabilities(t,a,b,c,d).mean(axis=0)
    np.testing.assert_allclose(x.mean(axis=0),predicted,atol=.018)


@pytest.mark.parametrize("model",["3PL","4PL"])
def test_recovery_pipeline_records_unconverged_fits_without_false_claims(model):
    params=_items(model)
    x,_=simulate_irt(120,params,seed=18)
    result=parameter_recovery(x,params,model,points=11,max_iter=2)
    assert len(result["recovery"])==len(params)*(3 if model=="3PL" else 4)
    summary=result["summary"]
    assert summary["n"]==120
    assert summary["k"]==5
    assert summary["boundary_parameter_count"]>=0
    assert summary["MAE"]>=0
    assert isinstance(summary["converged"],bool)


def test_recovery_rejects_misaligned_items():
    params=_items("2PL")
    x,_=simulate_irt(100,params)
    with pytest.raises(ValueError,match="sırası"):
        parameter_recovery(x,params.iloc[::-1].reset_index(drop=True),"2PL")
