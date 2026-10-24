import numpy as np
import pytest
from core.irt_python_mirt_parameters import python_mirt_binary_parameters
from core.irt_library_parameter_mapping import curves_from_normalized_table


@pytest.mark.parametrize("model", ["1PL", "2PL", "3PL", "4PL"])
def test_real_python_mirt_coefficients_feed_finite_icc_iif_tif(model):
    mirt = pytest.importorskip("mirt")
    x = mirt.simdata(model=model, n_persons=160, n_items=8, seed=2026)
    fit = mirt.fit_mirt(x, model=model)
    names = [f"Q{i}" for i in range(8)]
    table = python_mirt_binary_parameters(fit, model=model, item_names=names)
    assert table["Madde"].tolist() == names
    assert list(table.columns) == ["Madde", "a", "b", "c", "d"]
    assert len(table) == 8
    curves = curves_from_normalized_table(table, np.linspace(-4, 4, 41))
    assert curves["probability"].shape == (41, 8)
    assert curves["information"].shape == (41, 8)
    assert np.isfinite(curves["probability"]).all()
    assert np.isfinite(curves["information"]).all()
    assert np.all((curves["probability"] > 0) & (curves["probability"] < 1))
    assert np.all(curves["information"] >= 0)
    np.testing.assert_allclose(curves["test_information"],
                               curves["information"].sum(axis=1))
    if model == "1PL":
        np.testing.assert_allclose(table["a"], fit.coef()["discrimination"])
    if model in ("3PL", "4PL"):
        np.testing.assert_allclose(table["c"], fit.coef()["guessing"])
    if model == "4PL":
        np.testing.assert_allclose(table["d"], fit.coef()["upper"])


@pytest.mark.parametrize("model", ["1PL", "2PL", "3PL", "4PL"])
def test_library_curve_matches_independent_logistic_equation(model):
    """Independent NumPy formula, not the platform's curve implementation."""
    from scipy.special import expit
    mirt = pytest.importorskip("mirt")
    x = mirt.simdata(model=model, n_persons=160, n_items=8, seed=2026)
    fit = mirt.fit_mirt(x, model=model)
    table = python_mirt_binary_parameters(fit, model=model)
    theta = np.linspace(-3, 3, 23)
    a, b, c, d = (table[col].to_numpy(float) for col in ("a", "b", "c", "d"))
    expected_p = c[None, :] + (d - c)[None, :] * expit(
        a[None, :] * (theta[:, None] - b[None, :]))
    actual = curves_from_normalized_table(table, theta)
    np.testing.assert_allclose(actual["probability"], expected_p, rtol=1e-11, atol=1e-12)
    # Finite-difference Fisher information is independent of the analytic derivative.
    h = 1e-5
    p_plus = c[None, :] + (d-c)[None, :] * expit(
        a[None, :] * (theta[:, None] + h - b[None, :]))
    p_minus = c[None, :] + (d-c)[None, :] * expit(
        a[None, :] * (theta[:, None] - h - b[None, :]))
    derivative = (p_plus - p_minus) / (2*h)
    expected_info = derivative**2 / (expected_p*(1-expected_p))
    np.testing.assert_allclose(actual["information"], expected_info, rtol=1e-6, atol=1e-8)
