import numpy as np
import pandas as pd
from core.irt_simulation import simulate_irt
from core.irt_truth_scenarios import generating_items
from core.irt_stationarity import stationarity_diagnostics

def test_first_order_diagnostics_correct_parameter_count_and_signs():
    params=generating_items("4PL",5)     # identifiable: 20 <= 2**5-1
    x,_=simulate_irt(120,params,seed=91)
    params.loc[0,"a"]=4.
    details,summary=stationarity_diagnostics(x,params,"4PL",points=21)
    assert len(details)==20
    assert summary["parameters"]==20
    assert summary["n"]==120
    assert details.iloc[0]["Konum"]=="Üst sınır"
    slope=details.iloc[0]["Marjinal türev/kişi"]
    assert details.iloc[0]["KKT ihlali/kişi"]==max(-slope,0.)
    assert np.isfinite(details["KKT ihlali/kişi"]).all()

def test_three_pl_excludes_fixed_upper_asymptote():
    params=generating_items("3PL",4)     # identifiable: 12 <= 2**4-1
    x,_=simulate_irt(100,params,seed=92)
    details,summary=stationarity_diagnostics(x,params,"3PL",points=15)
    assert len(details)==12
    assert set(details["Parametre"])=={"a","b","c"}
