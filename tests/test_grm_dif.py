import numpy as np
import pandas as pd
import pytest
import json
from pathlib import Path
from scipy.integrate import quad
from scipy.special import expit
from scipy.stats import norm
from core import grm_dif as G
from scripts.grm_dif_validation import generate

FIXTURES = Path(__file__).parent / "fixtures" / "grm_dif"


@pytest.mark.parametrize("categories", [3, 5])
def test_category_probability_and_scores_against_independent_differences(categories):
    theta = np.array([-30., -2., .3, 3., 30.])
    a = np.array([.7, 1.6])
    b = np.array([np.linspace(-1.5, 1.2, categories - 1), np.linspace(-.8, 1.8, categories - 1)])
    lp, da, db, dt = G.category_terms(theta, a, b)
    np.testing.assert_allclose(np.exp(lp).sum(axis=2), 1, atol=1e-14)
    assert np.isfinite(lp).all() and np.isfinite(da).all() and np.isfinite(db).all()
    h = 1e-5
    for i in range(2):
        step = np.zeros_like(a); step[i] = h
        numeric = (G.category_terms(theta, a + step, b)[0] - G.category_terms(theta, a - step, b)[0]) / (2 * h)
        np.testing.assert_allclose(da[:, i], numeric[:, i], atol=1e-8, rtol=1e-6)
        for c in range(categories - 1):
            shift = np.zeros_like(b); shift[i, c] = h
            numeric = (G.category_terms(theta, a, b + shift)[0] - G.category_terms(theta, a, b - shift)[0]) / (2 * h)
            np.testing.assert_allclose(db[:, i, :, c], numeric[:, i], atol=1e-8, rtol=1e-6)
    numeric = (G.category_terms(theta + h, a, b)[0] - G.category_terms(theta - h, a, b)[0]) / (2 * h)
    np.testing.assert_allclose(dt, numeric, atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize("middle,size", [("b", 21), ("a", 19)])
def test_ordered_design_dimensions_and_physical_gradient_with_missingness(middle, size):
    rng = np.random.default_rng(17)
    x = rng.integers(0, 4, (220, 3)); x[::7, 1] = -1
    groups = np.repeat([0, 1], 110)
    parts = G.pattern_parts(x, groups, 4)
    d = G.Design(["anchor", middle, "ab"], 4)
    assert d.size == size
    a = np.array([[1., 1.2, .9], [1., 1.2, 1.4]])
    b = np.array([[[-1., 0., .9]] * 3, [[-1., 0., .9], [-.8, .2, 1.4], [-1., .1, 1.2]]])
    p = d.encode(a, b, -.4, 1.2)
    fn = lambda q: G.objective(q, d, parts, G._grid(41))
    score = fn(p)[1]
    for j in range(len(p)):
        step = np.zeros_like(p); step[j] = 2e-6
        numeric = (fn(p + step)[0] - fn(p - step)[0]) / 4e-6
        assert score[j] == pytest.approx(numeric, abs=2e-8, rel=2e-5)


def test_marginal_likelihood_against_adaptive_integral():
    d = G.Design(["anchor"] * 3, 3)
    a = np.array([[.8, 1.1, 1.4]] * 2)
    b = np.array([[[-1., .6], [-.7, .9], [-.4, 1.2]]] * 2)
    p = d.encode(a, b, -.4, 1.2)
    x = np.array([[0, 1, 2], [1, -1, 0], [2, 2, 1], [0, 0, 1]])
    g = np.array([0, 0, 1, 1])
    native = -G.objective(p, d, G.pattern_parts(x, g, 3), G._grid(81))[0] * len(x)
    reference = 0.
    for row, group in zip(x, g):
        def integrand(theta):
            survival = np.column_stack([np.ones(3), expit(a[group, :, None] * (theta - b[group])), np.zeros(3)])
            probs = survival[:, :-1] - survival[:, 1:]
            observed = row >= 0
            return np.prod(probs[np.arange(3)[observed], row[observed]]) * norm.pdf(theta, -.4 if group else 0., 1.2 if group else 1.)
        reference += np.log(quad(integrand, -np.inf, np.inf, epsabs=1e-11)[0])
    assert native == pytest.approx(reference, abs=2e-7)


@pytest.mark.parametrize("change,match", [
    ("nominal", "sıralı"), ("fraction", "tamsayı"), ("infinite", "Sonsuz"),
    ("sparse", "kategori"), ("duplicate", "benzersiz"), ("one_anchor", "iki sabit"),
])
def test_input_failure_modes(change, match):
    x, g, _ = generate(n_reference=200, n_focal=200)
    anchors = ["M1", "M2"]
    if change == "nominal": x = x.astype(str).replace({"0": "A", "1": "B", "2": "C"})
    elif change == "fraction": x = x.astype(float); x.iloc[0, 0] = .5
    elif change == "infinite": x = x.astype(float); x.iloc[0, 0] = np.inf
    elif change == "sparse": x.loc[g == "F", "M1"] = 1
    elif change == "duplicate": x.columns = ["M1", "M1", "M3", "M4", "M5", "M6"]
    elif change == "one_anchor": anchors = ["M1"]
    with pytest.raises(ValueError, match=match):
        G.validate_inputs(x, g, "R", "F", anchors)


def test_native_planted_threshold_and_slope_models():
    x, g, truth = generate()
    out = G.test_dif(x, g, "R", "F", ["M1", "M2", "M3", "M4"])
    t = out["table"].set_index("Madde")
    assert t.loc[["M1", "M2", "M3", "M4"], "p"].isna().all()
    assert t.loc["M5", "DIF işareti"] and t.loc["M6", "DIF işareti"]
    assert t.loc["M5", "sd (genel)"] == 3 and t.loc["M5", "sd (eşikler)"] == 2
    assert t.loc["M6", "sd (eğim | eşikler)"] == 1
    assert t.loc["M5", "q (BH, eşikler)"] < .05 and t.loc["M6", "q (BH, eğim | eşikler)"] < .05
    assert all(f["converged"] and f["cov"] is not None for fits in out["fits"].values() for f in fits.values())
    final = out["fits"][4]["H2"]
    a, b, mu, sd = final["design"].unpack(final["params"])
    assert a[1, 5] < a[0, 5] and np.mean(b[1, 4] - b[0, 4]) > .3
    assert abs(mu - truth["mu"]) < .3 and abs(sd - truth["sd"]) < .3


def test_probabilities_match_public_python_mirt_model():
    pytest.importorskip("mirt")
    from mirt.models.polytomous import GradedResponseModel
    model = GradedResponseModel(n_items=2, n_categories=3)
    a = np.array([.8, 1.7]); b = np.array([[-1., .7], [-.3, 1.6]])
    model.set_parameters(discrimination=a, thresholds=b)
    theta = np.linspace(-4, 4, 37)
    native = np.exp(G.category_terms(theta, a, b)[0])
    np.testing.assert_allclose(native, model.probability(theta[:, None]), atol=1e-12, rtol=1e-12)


def test_curve_and_effect_delta_gradients_include_linking_uncertainty():
    x, groups, _ = generate(n_reference=200, n_focal=200)
    inputs = G.validate_inputs(x, groups, "R", "F", ["M1", "M2", "M3", "M4"])
    d = G.Design(["anchor"] * 4 + ["ab", "ab"], 3)
    a = np.array([np.linspace(.9, 1.6, 6)] * 2); a[1, 5] -= .8
    b = np.array([[[-.9, .9]] * 6] * 2); b[1, 4] += .6
    p = d.encode(a, b, -.4, 1.2)
    fitted = dict(design=d, params=p, cov=np.eye(d.size) * .01)
    theta = np.array([-1.2, .1, 1.8])
    pr, pf, jac, _ = G._item_jacobian(fitted, 4, theta)
    numeric = np.zeros_like(jac)
    for j in range(d.size):
        step = np.zeros(d.size); step[j] = 1e-5
        plus = dict(fitted, params=p + step); minus = dict(fitted, params=p - step)
        r1, f1, _, _ = G._item_jacobian(plus, 4, theta)
        r0, f0, _, _ = G._item_jacobian(minus, 4, theta)
        numeric[:, :, j] = ((f1 - r1) - (f0 - r0)) / 2e-5
    np.testing.assert_allclose(jac, numeric, atol=1e-9)
    assert np.allclose((pf - pr).sum(axis=1), 0)
    # Independent numeric derivative of the focal-weighted signed effect.
    derivative = []
    for j in range(d.size):
        step = np.zeros(d.size); step[j] = 1e-5
        plus = G.signed_effect(dict(fitted, params=p + step), 4)["İşaretli etki (puan)"]
        minus = G.signed_effect(dict(fitted, params=p - step), 4)["İşaretli etki (puan)"]
        derivative.append((plus - minus) / 2e-5)
    effect = G.signed_effect(fitted, 4)
    assert effect["SH (etki)"] == pytest.approx(np.linalg.norm(derivative) * .1, rel=1e-7)
    assert effect["İşaretli etki (puan)"] < 0
    category, score = G.conditional_curves(fitted, 4, theta)
    assert len(category) == 9 and len(score) == 3 and np.isfinite(score.SH).all()
    assert category["%95 GA alt"].le(category["Olasılık farkı"]).all()


@pytest.mark.parametrize("failure", ["nonconvergence", "boundary", "information", "quadrature", "nested"])
def test_failure_never_becomes_evidence_of_no_dif(monkeypatch, failure):
    def fake(self, roles, *args, **kwargs):
        d = G.Design(roles, 3)
        p = d.encode(np.ones((2, 6)), np.array([[[-1., 1.]] * 6] * 2))
        return dict(design=d, params=p, cov=None if failure == "information" else np.eye(d.size),
                    converged=failure != "nonconvergence", at_bound=failure == "boundary",
                    quadrature_ok=failure != "quadrature",
                    loglik=-1000 + (-1 if failure == "nested" else 1) * sum(role != "anchor" for role in roles),
                    reason="Doğrulanamadı")
    monkeypatch.setattr(G.Fitter, "fit", fake)
    x, groups, _ = generate()
    out = G.test_dif(x, groups, "R", "F", list(x.columns[:4]))
    t = out["table"].iloc[4:]
    assert t.p.isna().all() and not t["DIF işareti"].any()
    assert t.Karar.eq("Çıkarım kullanılamıyor").all()
    assert out["parameters"].SH.isna().all()
    assert t["SH (etki)"].isna().all()


@pytest.mark.parametrize("condition", ["null", "both"])
def test_calibration_against_pinned_independent_python_mirt_em(condition):
    reference = json.loads((FIXTURES / f"python_mirt_{condition}.json").read_text())
    assert reference["external_version"] == "1.2.0"
    x, groups, _ = generate(reference["seed"], condition)
    inputs = G.validate_inputs(x, groups, "R", "F", list(x.columns[:4]))
    fitter = G.Fitter(inputs, reference["points"])
    roles = {"H0": ["anchor"] * 5 + ["ab"], "Hb": ["anchor"] * 4 + ["b", "ab"],
             "Ha": ["anchor"] * 4 + ["a", "ab"], "H2": ["anchor"] * 4 + ["ab", "ab"]}
    fitted = {}
    for r in reference["results"]:
        native = fitter.fit(roles[r["hypothesis"]])
        fitted[r["hypothesis"]] = native
        assert native["converged"] and native["quadrature_ok"] and native["cov"] is not None
        assert r["external_converged"] and native["design"].size == r["external_parameters"]
        a, b, mu, sd = native["design"].unpack(native["params"])
        assert native["loglik"] == pytest.approx(r["external_loglik"], abs=1e-3)
        np.testing.assert_allclose(a, r["external_a"], atol=.002, rtol=0)
        np.testing.assert_allclose(b, r["external_b"], atol=.002, rtol=0)
        assert mu == pytest.approx(r["external_focal_mean"], abs=.002)
        assert sd == pytest.approx(r["external_focal_sd"], abs=.002)
    for reduced, df in (("H0", 3), ("Hb", 1), ("Ha", 2)):
        assert fitted["H2"]["design"].size - fitted[reduced]["design"].size == df
    # Independent scalar second derivatives, not the analytic-score Jacobian.
    fit = fitted["H2"]; p = fit["params"]; info = np.linalg.inv(fit["cov"])
    fn = lambda q: G.objective(q, fit["design"], fitter.parts, fitter.grid)[0] * len(x)
    for v in np.random.default_rng(34).normal(size=(3, len(p))):
        v /= np.linalg.norm(v)
        h = 2e-4
        numeric = (fn(p + h * v) - 2 * fn(p) + fn(p - h * v)) / h ** 2
        assert v @ info @ v == pytest.approx(numeric, rel=1e-4, abs=.002)


def test_reference_group_reversal_preserves_the_common_response_metric():
    x, groups, _ = generate(n_reference=400, n_focal=300)
    anchors = list(x.columns[:4])
    original = G.test_dif(x, groups, "R", "F", anchors, points=61)
    reverse = G.test_dif(x, groups, "F", "R", anchors, points=61)
    f = original["fits"][4]["H2"]; r = reverse["fits"][4]["H2"]
    a, b, mu, sd = f["design"].unpack(f["params"])
    ar, br, mur, sdr = r["design"].unpack(r["params"])
    assert f["loglik"] == pytest.approx(r["loglik"], abs=1e-3)
    np.testing.assert_allclose(ar, a[::-1] * sd, atol=.003, rtol=0)
    np.testing.assert_allclose(br, (b[::-1] - mu) / sd, atol=.003, rtol=0)
    assert mur == pytest.approx(-mu / sd, abs=.003) and sdr == pytest.approx(1 / sd, abs=.003)


def test_missingness_uses_the_same_selected_rows_and_keeps_category_codes():
    x, groups, _ = generate(n_reference=200, n_focal=200)
    x = x.astype(float)
    x.loc[0] = np.nan; x.loc[1, "M1"] = np.nan; x.loc[201, "M1"] = np.nan
    x.loc[len(x)] = np.inf
    groups = pd.concat([groups, pd.Series(["other"])], ignore_index=True)
    inputs = G.validate_inputs(x, groups, "R", "F", ["M1", "M2", "M3", "M4"])
    assert inputs["sizes"] == [199, 200] and inputs["excluded_rows"] == 2
    assert inputs["missingness"][:, 0].tolist() == [1, 1]
    assert inputs["retained_rows"][0] == 1 and inputs["retained_rows"][-1] == 399
    assert inputs["x"][0, 0] == -1 and set(np.unique(inputs["x"])) == {-1, 0, 1, 2}


def test_hessian_and_quadrature_failures_are_not_hidden_by_optimizer_success():
    x, groups, _ = generate(n_reference=200, n_focal=200)
    inputs = G.validate_inputs(x, groups, "R", "F", list(x.columns[:4]))
    f = G.Fitter(inputs, points=21).fit(["anchor"] * 4 + ["ab", "ab"], max_iter=1)
    assert not f["converged"] and f["cov"] is None and f["reason"]
