import numpy as np
import pytest
from core.irt_truth_scenarios import SCENARIOS,generating_items
from core.irt_simulation import simulate_irt

@pytest.mark.parametrize("model",("3PL","4PL"))
@pytest.mark.parametrize("scenario",SCENARIOS)
def test_truth_regime_is_within_assumed_model_bounds(model,scenario):
    tab=generating_items(model,25,scenario)
    assert len(tab)==25
    assert tab["a"].between(.15,4.).all()
    assert tab["b"].between(-4.,4.).all()
    assert tab["c"].between(0.,.35).all()
    assert (tab["d"]>tab["c"]).all()
    if model=="3PL":
        assert np.all(tab["d"]==1.)
    else:
        assert tab["d"].between(.65,1.).all()
    x,_=simulate_irt(60,tab,seed=401)
    assert x.shape==(60,25)

def test_invalid_regime_rejected():
    with pytest.raises(ValueError,match="Unknown"):
        generating_items("3PL",25,"invented")
