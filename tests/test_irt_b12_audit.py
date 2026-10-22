"""Fast B12 audit tests; no expensive EM calibration required."""
import pandas as pd
import pytest
from core.irt_b12_audit import audit_fit


def _fit(model="4PL"):
    return {
        "model": model,
        "items": pd.DataFrame([
            {"Madde": "M01", "a": 1.3, "b": -.2, "c": .16, "d": .92},
            {"Madde": "M02", "a": 1.7, "b": .3, "c": .2, "d": 1.0},
        ]),
        "history": [-900., -880., -879.5],
        "n": 500, "iterations": 3, "converged": True,
        "convergence": {"criterion": "loglik"}, "a_upper": 4., "d_lower": .65,
    }


def test_4pl_detects_upper_bound_without_claiming_identifiability():
    detail, summary = audit_fit(_fit())
    assert len(detail) == 8
    assert detail.query("item == 'M02' and parameter == 'd'").iloc[0]["position"] == "upper_bound"
    assert summary["bound_parameters"] == 1
    assert not summary["inference_ready"]
    assert any("identification" in x for x in summary["warnings"])


def test_3pl_does_not_treat_fixed_d_as_free():
    detail, summary = audit_fit(_fit("3PL"))
    assert len(detail) == 6
    assert summary["bound_parameters"] == 0
    assert summary["inference_ready"]


def test_failed_convergence_and_likelihood_decrease():
    fit = _fit()
    fit.update(converged=False, history=[-880., -900.])
    _, result = audit_fit(fit)
    assert result["likelihood_decreases"] == 1
    assert not result["inference_ready"]


def test_invalid_parameter_and_tolerance_rejected():
    fit = _fit()
    fit["items"].loc[0, "c"] = .5
    detail, result = audit_fit(fit)
    assert result["invalid_parameters"] == 1
    assert not result["inference_ready"]
    with pytest.raises(ValueError):
        audit_fit(fit, bound_tol=0)
