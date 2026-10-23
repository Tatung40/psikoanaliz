import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.special import softmax
from scipy.stats import norm
from core import polytomous_dif as P
from scripts.polytomous_dif_validation import generate, ANCHORS

FIXTURES = Path(__file__).parent / "fixtures" / "polytomous_dif"


@pytest.mark.parametrize("model", P.MODELS)
def test_design_gradient_and_constraints_with_missingness(model):
    rng = np.random.default_rng(31)
    x = rng.integers(0, 4, (220, 3)); x[::7, 1] = -1
    groups = np.repeat([0, 1], 110)
    parts = P.pattern_parts(x, groups, 4)
    d = P.Design(model, ["anchor", "b", "ab"], 4)
    p = np.array([(lo + hi) / 2 for lo, hi in d.bounds]); p[-2:] = -.4, np.log(1.2)
    p[:d.latent] += rng.normal(0, .12, d.latent)
    fn = lambda q: P.objective(q, d, parts, P._grid(61))
    gradient = fn(p)[1]
    for j in range(d.size):
        step = np.zeros(d.size); step[j] = 2e-6
        assert gradient[j] == pytest.approx((fn(p + step)[0] - fn(p - step)[0]) / 4e-6, abs=2e-8)
    A, D, mu, sd = d.unpack(p)
    np.testing.assert_allclose(d.encode(A, D, mu, sd), p, atol=1e-14)
    lp = P.log_probabilities(np.array([-100., 0., 100.]), A[0], D[0])
    assert np.isfinite(lp).all()
    np.testing.assert_allclose(np.exp(lp).sum(axis=2), 1, atol=1e-13)
    if model == "RSM":
        steps = -np.diff(D, axis=2); centered = steps - steps.mean(axis=2, keepdims=True)
        np.testing.assert_allclose(centered, np.broadcast_to(centered[0, 0], centered.shape), atol=1e-14)
    if model in ("PCM", "RSM"): np.testing.assert_equal(A, np.broadcast_to(np.arange(4), A.shape))
    if model == "NRM": assert np.all(A[:, :, 0] == 0) and np.all(D[:, :, 0] == 0)


@pytest.mark.parametrize("model", P.MODELS)
def test_fixed_parameter_likelihood_against_adaptive_integral(model):
    _, _, truth = generate(model)
    d = P.Design(model, ["anchor"] * 4 + ["ab"] * 2, 3)
    p = d.encode(truth["A"], truth["D"], -.4, 1.2)
    x = np.array([[0, 1, 2, -1, 0, 1], [1, -1, 0, 2, 1, 0], [2, 2, 1, 1, 0, 2], [0, 0, 1, 1, 1, 2]])
    groups = np.array([0, 0, 1, 1])
    native = -P.objective(p, d, P.pattern_parts(x, groups, 3), P._grid(101))[0] * len(x)
    reference = 0.
    for row, g in zip(x, groups):
        def integrand(theta):
            pr = softmax(truth["A"][g] * theta + truth["D"][g], axis=1)
            observed = row >= 0
            return np.prod(pr[np.arange(6)[observed], row[observed]]) * norm.pdf(theta, -.4 if g else 0., 1.2 if g else 1.)
        reference += np.log(quad(integrand, -np.inf, np.inf, epsabs=1e-11)[0])
    assert native == pytest.approx(reference, abs=2e-7)


@pytest.mark.parametrize("model", P.MODELS)
def test_planted_dif_and_effect_jacobian(model):
    x, g, _ = generate(model)
    out = P.test_dif(x, g, "R", "F", ANCHORS, model=model)
    t = out["table"].set_index("Madde")
    assert t.loc[ANCHORS, "p"].isna().all()
    assert t.loc[["M5", "M6"], "p"].notna().all(), t.loc[["M5", "M6"], "Not"].tolist()
    assert t.loc[["M5", "M6"], "DIF işareti"].all()
    assert t.loc["M5", "sd (genel)"] == {"PCM": 2, "GPCM": 3, "RSM": 1, "NRM": 4}[model]
    fit = out["fits"][4]["H2"]; d, p = fit["design"], fit["params"]
    theta = np.array([-1., .2, 1.])
    pr, pf, jac, dt = P.item_jacobian(fit, 4, theta)
    for j in range(d.size):
        step = np.zeros(d.size); step[j] = 1e-5
        plus, minus = dict(fit, params=p + step), dict(fit, params=p - step)
        def diff(f):
            r, s, _, _ = P.item_jacobian(f, 4, theta)
            return s - r
        np.testing.assert_allclose(jac[:, :, j], (diff(plus) - diff(minus)) / 2e-5, atol=1e-9)
    categories, score = P.conditional_curves(fit, 4)
    assert categories["SH"].notna().all()
    assert score.empty == (model == "NRM")
    assert (out["effects"].SH > 0).all()


@pytest.mark.parametrize("change", ["sparse", "infinite", "fraction", "labels", "anchor", "group", "duplicate"])
def test_input_failure_modes(change):
    x, g, _ = generate(n_reference=200, n_focal=200)
    anchors = ANCHORS
    if change == "sparse": x.loc[g == "F", "M1"] = 0
    if change == "infinite": x = x.astype(float); x.iloc[0, 0] = np.inf
    if change == "fraction": x = x.astype(float); x.iloc[0, 0] = .5
    if change == "labels": x = x.astype(str).replace({"0": "A", "1": "B", "2": "C"})
    if change == "anchor": anchors = ["M1"]
    if change == "group": g.name = "M1"
    if change == "duplicate": x.columns = ["M1", "M1", "M3", "M4", "M5", "M6"]
    with pytest.raises(ValueError): P.validate_inputs(x, g, "R", "F", anchors)


def test_nominal_explicit_labels_without_order_and_unknown_rejection():
    x, g, _ = generate("NRM", n_reference=200, n_focal=200)
    nominal = x.replace({0: "Z", 1: "A", 2: "K"})
    inputs = P.validate_inputs(nominal, g, "R", "F", ANCHORS, "NRM", category_labels=["Z", "A", "K"])
    np.testing.assert_equal(inputs["x"], x.to_numpy())
    nominal.iloc[0, 0] = "UNKNOWN"
    with pytest.raises(ValueError, match="etiket"): P.validate_inputs(nominal, g, "R", "F", ANCHORS, "NRM", category_labels=["Z", "A", "K"])


@pytest.mark.parametrize("model", P.MODELS)
@pytest.mark.parametrize("condition", ["null", "both"])
def test_pinned_independent_calibration_and_degrees_of_freedom(model, condition):
    prefix = "independent" if model == "RSM" else "python_mirt"
    fixture = json.loads((FIXTURES / f"{prefix}_{model}_{condition}.json").read_text())
    x, g, _ = generate(model, fixture["seed"], condition)
    fitter = P.Fitter(P.validate_inputs(x, g, "R", "F", ANCHORS, model), fixture["points"])
    start = None
    for r in fixture["results"]:
        roles = ["anchor"] * 4 + [{"H0": "anchor", "Hb": "b", "Ha": "a", "H2": "ab"}[r["hypothesis"]], "ab"]
        f = fitter.fit(roles, start); start = f
        assert r["external_converged"] and f["converged"] and f["cov"] is not None and f["quadrature_ok"]
        assert f["design"].size == r["external_parameters"]
        assert f["loglik"] == pytest.approx(r["external_loglik"], abs=.002)
        A, D, mu, sd = f["design"].unpack(f["params"])
        np.testing.assert_allclose(A, r["external_A"], atol=.002)
        np.testing.assert_allclose(D, r["external_D"], atol=.002)
        assert mu == pytest.approx(r["external_focal_mean"], abs=.002)
        assert sd == pytest.approx(r["external_focal_sd"], abs=.002)


@pytest.mark.parametrize("failure", ["convergence", "bound", "information", "quadrature", "negative_lr"])
def test_invalid_fit_never_becomes_absence_of_dif(monkeypatch, failure):
    class FakeFitter:
        def __init__(self, inputs, points):
            self.inputs, self.optimizations, self.cache = inputs, 0, {}
        def fit(self, roles, start_fit=None):
            d = P.Design(self.inputs["model"], roles, 3)
            A = np.broadcast_to(np.arange(3), (2, len(roles), 3)).copy()
            D = np.zeros_like(A)
            params = d.encode(A, D)
            bad = any(r != "anchor" for r in roles)
            f = dict(design=d, params=params, loglik=-100 + (-1 if failure == "negative_lr" else 1) * d.size,
                converged=not (failure == "convergence" and bad), at_bound=failure == "bound" and bad,
                cov=None if failure == "information" and bad else np.eye(d.size),
                quadrature_ok=not (failure == "quadrature" and bad), reason=failure if bad else "")
            self.optimizations += 1; self.cache[tuple(roles)] = f
            return f
    monkeypatch.setattr(P, "Fitter", FakeFitter)
    x, g, _ = generate("GPCM", n_reference=200, n_focal=200)
    out = P.test_dif(x, g, "R", "F", ANCHORS, model="GPCM")
    assert out["table"].p.isna().all() and not out["table"]["DIF işareti"].any()
    assert out["table"].loc[4:, "Karar"].eq("Çıkarım kullanılamıyor").all()
    assert out["parameters"].SH.isna().all() and out["effects"].SH.isna().all()


def test_actual_nonconvergence_is_detected():
    x, g, _ = generate("PCM", n_reference=200, n_focal=200)
    fitter = P.Fitter(P.validate_inputs(x, g, "R", "F", ANCHORS), 61)
    fit = fitter.fit(["anchor"] * 4 + ["ab"] * 2, max_iter=1)
    assert not fit["converged"] and fit["cov"] is None


@pytest.mark.parametrize("model", P.MODELS)
def test_group_reversal_invariant_likelihood_and_latent_metric(model):
    x, g, _ = generate(model, n_reference=400, n_focal=400)
    roles = ["anchor"] * 4 + ["ab"] * 2
    fits = [P.Fitter(P.validate_inputs(x, g, r, f, ANCHORS, model), 81).fit(roles) for r, f in (("R", "F"), ("F", "R"))]
    assert all(f["converged"] and f["quadrature_ok"] for f in fits)
    first, reverse = [f["design"].unpack(f["params"]) for f in fits]
    A, D, mu, sd = first; RA, RD, rm, rs = reverse
    # PCM/RSM fix slopes to 1, so changing reference unit variance is not a
    # permitted reparameterization. Only free-slope models have this symmetry.
    if model in ("GPCM", "NRM"):
        assert fits[0]["loglik"] == pytest.approx(fits[1]["loglik"], abs=.002)
        np.testing.assert_allclose(RA, A[::-1] * sd, atol=.003)
        np.testing.assert_allclose(RD, D[::-1] + A[::-1] * mu, atol=.003)
        assert rm == pytest.approx(-mu / sd, abs=.003)
        assert rs == pytest.approx(1 / sd, abs=.003)


def test_nominal_permutation_preserves_likelihood_and_label_probabilities():
    x, g, _ = generate("NRM")
    roles = ["anchor"] * 4 + ["ab"] * 2
    original = P.Fitter(P.validate_inputs(x, g, "R", "F", ANCHORS, "NRM"), 81).fit(roles)
    # Move baseline and reverse one category contrast: no rank/matching score.
    order = np.array([2, 0, 1]); inverse = np.argsort(order)
    relabeled = x.map(lambda value: inverse[value])
    altered = P.Fitter(P.validate_inputs(relabeled, g, "R", "F", ANCHORS, "NRM"), 81).fit(roles)
    assert original["converged"] and altered["converged"]
    assert altered["loglik"] == pytest.approx(original["loglik"], abs=.002)
    A, D, mu, sd = original["design"].unpack(original["params"])
    RA, RD, rm, rs = altered["design"].unpack(altered["params"])
    expected_A = A[:, :, order] - A[:, :, order[:1]]
    expected_D = D[:, :, order] - D[:, :, order[:1]]
    sign = 1 if expected_A[0, 0, 1] > 0 else -1
    np.testing.assert_allclose(RA, sign * expected_A, atol=.003)
    np.testing.assert_allclose(RD, expected_D, atol=.003)
    assert rm == pytest.approx(sign * mu, abs=.003) and rs == pytest.approx(sd, abs=.003)


def test_pcm_steps_are_not_sorted_and_missing_subset_is_common():
    x, g, truth = generate("PCM", n_reference=200, n_focal=200)
    d = P.Design("PCM", ["anchor"] * 4 + ["ab"] * 2, 3)
    p = d.encode(truth["A"], truth["D"])
    assert (np.diff(d.physical(p)[1], axis=2) < 0).all()
    x = x.astype(float); x.iloc[0, :] = np.nan; x.iloc[1, 0] = np.nan
    x = pd.concat([x, x.iloc[:2]], ignore_index=True); g = pd.concat([g, pd.Series(["OTHER", None])], ignore_index=True)
    inputs = P.validate_inputs(x, g, "R", "F", ANCHORS)
    assert inputs["excluded_rows"] == 3 and inputs["missingness"].sum() == 1
    assert 0 not in inputs["retained_rows"] and inputs["x"][0, 0] == -1


@pytest.mark.parametrize("model", P.MODELS)
def test_public_model_probabilities_with_four_categories(model):
    from mirt.models import PartialCreditModel, GeneralizedPartialCredit, RatingScaleModel, NominalResponseModel
    external = {"PCM": PartialCreditModel, "GPCM": GeneralizedPartialCredit,
                "RSM": RatingScaleModel, "NRM": NominalResponseModel}[model](3, 4)
    d = P.Design(model, ["anchor"] * 3, 4)
    p = np.array([(lo + hi) / 2 for lo, hi in d.bounds])
    p[:d.latent] += np.random.default_rng(45).normal(0, .15, d.latent)
    a, b, tau = d.physical(p)
    if model == "NRM":
        external.set_parameters(slopes=np.column_stack([np.zeros(3), a[0]]), intercepts=np.column_stack([np.zeros(3), b[0]]))
    elif model == "RSM": external.set_parameters(difficulty=b[0, :, 0], thresholds=tau)
    elif model == "PCM": external.set_parameters(steps=b[0])
    else: external.set_parameters(discrimination=a[0, :, 0], steps=b[0])
    A, D, _, _ = d.unpack(p); theta = np.array([-3., -.2, 1., 3.])
    np.testing.assert_allclose(np.exp(P.log_probabilities(theta, A[0], D[0])), external.probability(theta), atol=1e-12)


@pytest.mark.parametrize("model", ["GPCM", "NRM"])
def test_weighted_effect_uncertainty_includes_latent_distribution_derivatives(model):
    _, _, truth = generate(model)
    d = P.Design(model, ["anchor"] * 4 + ["ab"] * 2, 3)
    p = d.encode(truth["A"], truth["D"], -.4, 1.2)
    fit = dict(design=d, params=p, cov=np.eye(d.size))
    analytical = P.weighted_effects(fit, 4)
    z, lw = P._grid(81); w = np.exp(lw)
    def value(params):
        A, D, mu, sd = d.unpack(params)
        pr = np.exp(P.log_probabilities(mu + sd * z, A[0, 4:5], D[0, 4:5]))[:, 0]
        pf = np.exp(P.log_probabilities(mu + sd * z, A[1, 4:5], D[1, 4:5]))[:, 0]
        v = w @ (pf - pr)
        return v if model == "NRM" else np.array([v @ np.arange(3)])
    gradient = []
    for j in range(d.size):
        step = np.zeros(d.size); step[j] = 2e-6
        gradient.append((value(p + step) - value(p - step)) / 4e-6)
    gradient = np.array(gradient).T
    np.testing.assert_allclose(analytical.SH, np.sqrt((gradient ** 2).sum(axis=1)), atol=2e-9)
