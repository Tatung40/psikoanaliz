import numpy as np
import pandas as pd
import pytest
from core.irt_simulation import simulate_irt, parameter_recovery, repeated_recovery
from core.irt_multistart import fit_irt_multistart

def _truth(model):
    return pd.DataFrame({
        "Madde":[f"M{i+1}" for i in range(5)],
        "a":[.8,1.1,1.5,1.,1.3],
        "b":[-1.,-.4,0.,.4,1.],
        "c":[.07,.1,.15,.12,.18],
        "d":[1.]*5 if model=="3PL" else [.91,.93,.95,.92,.96]})

@pytest.mark.parametrize("model",["3PL","4PL"])
def test_multistart_selection_and_parameter_support(model):
    true=_truth(model)
    x,_=simulate_irt(180,true,seed=761)
    audit=fit_irt_multistart(x,model=model,starts=(0.,-1.,1.),
                             points=15,max_iter=18)
    successful=audit["candidates"]["Başarılı"]
    candidates=audit["candidates"][successful]
    assert len(candidates)>=1
    if candidates["Yakınsama"].any():
        pool=candidates[candidates["Yakınsama"]]
    else:
        pool=candidates
    assert audit["fit"]["loglik"]==pytest.approx(pool["logLik"].max(),abs=1e-9)
    est=audit["fit"]["items"]
    assert est["a"].between(.15,4.).all()
    assert est["b"].between(-4.,4.).all()
    assert est["c"].between(0.,.35).all()
    assert est["d"].between(.65,1.).all()
    assert np.isfinite(est[["a","b","c","d"]].to_numpy()).all()
    assert audit["fit"]["aic"]>0
    assert audit["fit"]["bic"]>0

@pytest.mark.parametrize("model",["3PL","4PL"])
def test_recovery_tracks_three_starts_and_does_not_force_convergence(model):
    truth=_truth(model)
    x,_=simulate_irt(120,truth,seed=11)
    result=parameter_recovery(x,truth,model,
                              starts=(0.,-1.,1.),points=11,max_iter=2)
    assert result["summary"]["start_count"]==3
    assert isinstance(result["summary"]["converged"],bool)
    assert result["summary"]["boundary_parameter_count"]>=0
    assert np.isfinite(result["recovery"]["Hata"]).all()

def test_multiple_recovery_records_start_count():
    truth=_truth("3PL")
    outcome=repeated_recovery(truth,models=("3PL",),
                              sample_sizes=(100,),repetitions=1,
                              starts=(0.,-1.,1.),points=11,max_iter=2)
    assert outcome["runs"]["Başlangıç sayısı"].iloc[0]==3
