from types import SimpleNamespace
import pytest
from core.irt_library_diagnostics import library_fit_diagnostics


def test_diagnostics_from_real_library_fit():
    mirt = pytest.importorskip("mirt")
    x = mirt.simdata(model="2PL", n_persons=100, n_items=8, seed=2026)
    fit = mirt.fit_mirt(x, model="2PL")
    result = library_fit_diagnostics(fit)
    assert result["n_parameters"] == 16
    assert result["n_observations"] == 100
    assert result["iterations"] >= 0
    assert isinstance(result["converged"], bool)


def test_missing_required_field_fails_closed():
    with pytest.raises(ValueError, match="missing fields"):
        library_fit_diagnostics(SimpleNamespace(converged=True))


def test_nonfinite_information_criterion_fails_closed():
    fit = SimpleNamespace(converged=True, n_iterations=2,
        log_likelihood=-50., aic=float("nan"), bic=120.,
        n_parameters=4, n_observations=100)
    with pytest.raises(ValueError, match="Nonfinite"):
        library_fit_diagnostics(fit)
