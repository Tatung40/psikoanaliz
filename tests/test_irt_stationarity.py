import numpy as np
import pandas as pd
from core.irt_simulation import simulate_irt
from core.irt_truth_scenarios import generating_items
from core.irt_stationarity import stationarity_diagnostics

def test_first_order_diagnostics_correct_parameter_count_and_signs():
    params=generating_items("4PL",4)
    x,_=simulate_irt(120,params,seed=91)
    params.loc[0,"a"]=4.
    details,summary=stationarity_diagnostics(x,params,"4PL",points=21)
    assert len(details)==16
    assert summary["parameters"]==16
    assert summary["n"]==120
    assert details.iloc[0]["Konum"]=="Üst sınır"
    slope=details.iloc[0]["Marjinal türev/kişi"]
    assert details.iloc[0]["KKT ihlali/kişi"]==max(-slope,0.)
    assert np.isfinite(details["KKT ihlali/kişi"]).all()

def test_three_pl_excludes_fixed_upper_asymptote():
    params=generating_items("3PL",3)
    x,_=simulate_irt(100,params,seed=92)
    details,summary=stationarity_diagnostics(x,params,"3PL",points=15)
    assert len(details)==9
    assert set(details["Parametre"])=={"a","b","c"}
