import numpy as np
import pandas as pd
import pytest
from core.irt import probabilities,information,fit_irt
from core.irt_simulation import simulate_irt
from core.irt_truth_scenarios import generating_items
from core.irt_quadrature_validation import marginal_loglik_at_fixed_parameters


@pytest.mark.parametrize("model",("3PL","4PL"))
def test_information_matches_independent_theta_difference(model):
    truth=generating_items(model,5)
    a,b,c,d=[truth[q].to_numpy(float) for q in ("a","b","c","d")]
    theta=np.linspace(-2.,2.,13)
    eps=1e-5
    expected=probabilities(theta,a,b,c,d)
    assert (expected>=c[None,:]).all()
    assert (expected<=d[None,:]).all()
    deriv=(probabilities(theta+eps,a,b,c,d)
           -probabilities(theta-eps,a,b,c,d))/(2*eps)
    benchmark=deriv**2/(expected*(1-expected))
    np.testing.assert_allclose(information(theta,a,b,c,d),
                               benchmark,rtol=1e-8,atol=1e-9)
    assert (deriv>0).all()


@pytest.mark.parametrize("model",("3PL","4PL"))
def test_final_marginal_likelihood_matches_independent_evaluator(model):
    truth=generating_items(model,4 if model=="3PL" else 5)
    x,_=simulate_irt(125,truth,seed=82)
    fit=fit_irt(x,model=model,points=15,max_iter=4)
    independent=marginal_loglik_at_fixed_parameters(
        x,fit["items"],points=15)
    assert fit["loglik"]==pytest.approx(independent,abs=2e-7)
    assert fit["param_count"]==len(truth)*(3 if model=="3PL" else 4)
    assert fit["aic"]==pytest.approx(-2*fit["loglik"]+2*fit["param_count"])
    assert fit["bic"]==pytest.approx(
        -2*fit["loglik"]+np.log(len(x))*fit["param_count"])
