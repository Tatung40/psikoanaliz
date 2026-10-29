"""Retained native estimators reproduce the Python mirt library on the same model (cross-validation).

Decision record (validation/COVERAGE_MATRIX_2026-10.md): the native Rasch MML and fixed-anchor DIF
estimators are kept because they provide what the library route does not (MAR missing responses,
WLE/infit/outfit pipeline; validated p/q, Wald CIs, uniform/non-uniform splits, purification). Their
likelihood maxima must equal the library's under the same model, design and quadrature.
Predeclared tolerances: Rasch reparameterization 1e-4 (sigma, b), log-likelihood 1e-5; DIF LR 1e-3
chi-square units (<0.03% of the chi2(1) critical value), from optimizer stopping rules only.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mirt")
from core import example_datasets as ex
from core.irt_python_mirt_adapter import fit_python_mirt


def test_rasch_mml_equals_library_common_discrimination_1pl():
    """theta ~ N(0, sigma^2), P = logistic(theta - b)  ==  theta ~ N(0, 1), P = logistic(a(theta - b'))
    with sigma = a and b = a b'; both maximize the same marginal likelihood."""
    from core.rasch import fit_rasch
    x = pd.read_csv("tests/fixtures/rasch/data_rasch_main.csv").dropna().astype(int)
    native = fit_rasch(x)
    library = fit_python_mirt(x.to_numpy(), model="1PL", common_discrimination=True,
                              tol=1e-8, max_iter=5000, n_quadpts=61).fit
    coef = library.coef()
    a = float(coef["discrimination"].iloc[0])
    assert np.ptp(coef["discrimination"].to_numpy()) == 0
    assert native["sd"] == pytest.approx(a, abs=1e-4)
    np.testing.assert_allclose(np.asarray(native["b"], float), a * coef["difficulty"].to_numpy(), atol=1e-4)
    assert native["loglik"] == pytest.approx(library.log_likelihood, abs=1e-5)


def test_library_route_cannot_replace_native_rasch_for_missing_responses():
    x = pd.read_csv("tests/fixtures/rasch/data_rasch_main.csv").astype(float)
    x.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="Missing"):
        fit_python_mirt(x.to_numpy(), model="1PL", common_discrimination=True)


def _dif_example(model):
    spec = next(d for d in ex.registry().values() if d.family == "dif" and d.model == model and d.level == 1)
    return ex.materialize(spec.id)


def _library_lr(bundle, model, families):
    from core import python_mirt_dif as P
    items = bundle["items"]
    table = P.analyze(bundle["data"], items=items, group=bundle["group"], reference="Referans", focal="Odak",
                      model=model, anchors=items[:2], studied=items[2:4], parameters=families,
                      n_quadpts=61, tol=1e-8, max_iter=5000)["tables"]["DIF"]
    return table.set_index("Item")["LR"]


@pytest.mark.parametrize("model", ["2PL", "GRM", "PCM", "GPCM"])
def test_fixed_anchor_dif_omnibus_lr_equals_library_multigroup(model):
    bundle = _dif_example(model)
    items = bundle["items"]
    x, g = bundle["data"][items], bundle["data"][bundle["group"]]
    if model == "2PL":
        from core.irt_dif import dif_analysis
        table = dif_analysis(x, g, "Referans", "Odak", model="2PL", purify=False, fixed_anchors=items[:2],
                             points=61, min_group=100)["table"]
        native = table.set_index("Madde")["LR χ² genel (sd=2)"]
    elif model == "GRM":
        from core.grm_dif import test_dif
        native = test_dif(x, g, "Referans", "Odak", items[:2], points=61)["table"].set_index("Madde")["LR (genel)"]
    else:
        from core.polytomous_dif import test_dif
        native = test_dif(x, g, "Referans", "Odak", items[:2], model=model, categories=4,
                          points=61)["table"].set_index("Madde")["LR (genel)"]
    families = ("intercepts",) if model == "PCM" else ("discrimination", "intercepts")
    library = _library_lr(bundle, model, families)
    np.testing.assert_allclose(native.loc[library.index].to_numpy(float), library.to_numpy(float), rtol=0, atol=1e-3)
