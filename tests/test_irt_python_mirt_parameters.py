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
