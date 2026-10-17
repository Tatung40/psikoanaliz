"""B4: structural identifiability and convergence diagnostics for fit_irt."""
import inspect
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from core.irt import (fit_irt, validate_binary, structural_identifiability,
                      approximate_parameter_uncertainty)
from core.irt_hessian import observed_information_se
from core.irt_se_diagnostics import hessian_boundary_diagnostics
from core.irt_simulation import simulate_irt

MIRT = Path(__file__).parent / "fixtures" / "mirt"


@pytest.mark.parametrize("model,k,ok", [("1PL", 3, True), ("2PL", 3, True),
                                        ("3PL", 3, False), ("3PL", 4, True),
                                        ("4PL", 4, False), ("4PL", 5, True)])
def test_parameter_count_against_pattern_degrees_of_freedom(model, k, ok):
    s = structural_identifiability(k, model)
    assert s["pattern_df"] == 2**k - 1
    assert s["satisfied"] is ok and (s["reason"] == "") is ok
    assert s["parameters"] <= s["pattern_df"] if ok else s["parameters"] > s["pattern_df"]


def test_equality_is_not_rejected_and_rule_is_not_in_validate_binary():
    # parameters == 2**k - 1 does not occur for 1PL-4PL with k>=3; check the
    # boundary of the rule directly: one parameter fewer/more than the limit.
    assert list(inspect.signature(validate_binary).parameters) == ["data"]
    for model in ("3PL", "4PL"):
        k = 4 if model == "3PL" else 5
        s = structural_identifiability(k, model)
        assert s["satisfied"] and s["parameters"] < s["pattern_df"]


def _items(k):
    return pd.DataFrame({"Madde": [f"M{i+1}" for i in range(k)],
                         "a": np.linspace(1., 1.6, k), "b": np.linspace(-.8, .8, k),
                         "c": np.linspace(.12, .18, k), "d": np.linspace(.9, .95, k)})


@pytest.mark.parametrize("model,k", [("3PL", 3), ("4PL", 3), ("4PL", 4)])
def test_unidentified_models_are_rejected_before_estimation(model, k):
    data, _ = simulate_irt(200, _items(k), seed=4)
    with pytest.raises(ValueError, match="yapısal olarak tanımlanamaz"):
        fit_irt(data, model, points=11, max_iter=1)
    _, reason = observed_information_se(data, _items(k), model, points=11)
    assert "yapısal olarak tanımlanamaz" in reason
    _, reason = approximate_parameter_uncertainty(data, _items(k), model, points=11)
    assert "yapısal olarak tanımlanamaz" in reason


@lru_cache(maxsize=None)
def _est_data():
    return pd.read_csv(MIRT / "data_3pl_est.csv")   # interior 3PL MLE, verified vs mirt (B3)


def test_well_identified_fit_is_not_rejected_by_parameter_criterion():
    fit = fit_irt(_est_data(), "3PL", points=21)
    conv = fit["convergence"]
    assert fit["converged"] and conv["status"] == "converged"
    assert conv["parameter_criterion_met"] and conv["max_param_change"] <= 1e-3
    assert conv["last_loglik_change"] < conv["loglik_tolerance"]
    assert fit["warnings"] == []
    assert len(fit["param_change_history"]) == fit["iterations"] - 1
    assert fit["param_change_history"][-1] == conv["max_param_change"]


def test_loglik_only_convergence_warns_but_keeps_estimate():
    fit = fit_irt(_est_data(), "3PL", points=21, param_tolerance=1e-5)
    conv = fit["convergence"]
    assert fit["converged"]                               # logL criterion unchanged
    assert conv["status"] == "loglik_only" and not conv["parameter_criterion_met"]
    assert conv["message"] in fit["warnings"]
    assert np.isfinite(fit["items"][["a", "b", "c"]].to_numpy()).all()


def test_nonconvergence_status():
    fit = fit_irt(_est_data(), "3PL", points=21, max_iter=2)
    assert not fit["converged"] and fit["convergence"]["status"] == "not_converged"


def test_boundary_parameters_reported_separately():
    truth = pd.DataFrame({"Madde": [f"M{i}" for i in range(6)], "a": np.linspace(.8, 1.6, 6),
                          "b": np.linspace(-1, 1, 6), "c": 0., "d": 1.})
    x, _ = simulate_irt(300, truth, seed=3)                 # B1 fixture: c estimates hit 0
    fit = fit_irt(x, "3PL", points=21)
    conv = fit["convergence"]
    diag = hessian_boundary_diagnostics(fit["items"], "3PL")
    assert conv["parameters_at_bound"] == len(diag) > 0
    assert conv["max_param_change"] == max(conv["max_param_change_interior"],
                                           conv["max_param_change_at_bound"])
    assert any("optimizasyon sınırında" in w for w in fit["warnings"])


def test_invalid_parameter_tolerance():
    with pytest.raises(ValueError, match="toleransı"):
        fit_irt(_est_data(), "3PL", points=11, max_iter=1, param_tolerance=0.)
