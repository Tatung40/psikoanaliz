import numpy as np
import pandas as pd
from core.irt_simulation import simulate_irt
from core.irt_truth_scenarios import generating_items
from core.irt_quadrature_validation import (
    marginal_loglik_at_fixed_parameters,quadrature_sensitivity)

def test_fixed_parameter_likelihood_finite_both_models():
    for model in ("3PL","4PL"):
        truth=generating_items(model,4)
        x,_=simulate_irt(90,truth,seed=19)
        val=marginal_loglik_at_fixed_parameters(x,truth,points=31)
        assert np.isfinite(val) and val<0

def test_quadrature_comparison_uses_shared_reference_grid():
    truth=generating_items("3PL",4)      # 3PL with k=3 is structurally unidentified
    x,_=simulate_irt(90,truth,seed=19)
    result=quadrature_sensitivity(
        x,"3PL",fit_points=(11,15),reference_points=21,
        max_iter=2,starts=(0.,))
    assert len(result)==2
    assert result["Karşılaştırma düğümü"].eq(21).all()
    assert np.isfinite(result["Ortak ağ logLik"]).all()
    assert result.loc[result["Fit düğümü"]==15,
                      "Ortak ağ referansa parametre farkı ort."].iloc[0]==0.
