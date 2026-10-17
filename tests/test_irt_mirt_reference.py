"""B3: 3PL/4PL comparison with R mirt 1.48 on matched fixtures.

Reference values: tests/fixtures/mirt/mirt_results.csv, produced by
scripts/mirt_reference_3pl4pl.R (provenance in mirt_session.txt). Fixtures
come from scripts/make_mirt_fixtures.py. Only methodologically equivalent
quantities are compared:

* Parameterization: mirt P = g + (u-g)/(1+exp(-(a1*theta+d))), g and u on
  the logit scale internally; Python a=a1, b=-d/a1, c=g, d=u.
* Integration: identical 21-node Gauss-Hermite grid in both programs
  (mirt technical customTheta/customPriorFun).
* Derivatives: mirt quantities are in its internal space (a1, d, logit g,
  logit u). They are mapped with K = d(theta_mirt)/d(theta_python) and, for
  Hessians away from a stationary point, the second-order term
  sum_k g_k * d2 theta_mirt_k (omitting it changes the matrix by 10-30%).
* mirt's SE.type numerical Hessian (R/utils.R::BL.LL) re-weights the nodes by
  dnorm(theta)/sum and ignores customPriorFun; it is therefore compared on
  mirt's default 61-node grid, where logLik uses the same weights.

Tolerances (relative to the largest absolute entry) come from measured
agreement and from mirt's own numerical spread (Richardson vs central):
gradient: observed <=2.7e-9, mirt spread ~4e-9 -> 5e-8; Hessian: observed
<=6.6e-8, mirt spread <=1.0e-6 -> 1e-5; SE: observed <=7.7e-8 -> 1e-5.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from numpy.polynomial.hermite import hermgauss
from core.irt import fit_irt, probabilities
from core.irt_hessian import (_analytic_marginal_score, _observed_information,
                              marginal_score, observed_information_se)
from core.irt_quadrature_validation import marginal_loglik_at_fixed_parameters

FIX = Path(__file__).parent / "fixtures" / "mirt"
RESULTS = FIX / "mirt_results.csv"
if not RESULTS.exists():
    pytest.skip("R mirt reference values missing: run scripts/mirt_reference_3pl4pl.R "
                "(open validation task; no reference values are invented).",
                allow_module_level=True)

GRADIENT_RTOL = 5e-8
HESSIAN_RTOL = 1e-5
SE_RTOL = 1e-5
POINTS = 21
ACTIVE = {"3PL": ["a", "b", "c"], "4PL": ["a", "b", "c", "d"]}
CASES = pd.read_csv(FIX / "cases.csv")
FIXED = [c for c in CASES["case"] if not c.endswith("_est")]


@lru_cache(maxsize=None)
def _results():
    return pd.read_csv(RESULTS, keep_default_na=False, na_values=["NA"])


@lru_cache(maxsize=None)
def _case(case):
    row = CASES.set_index("case").loc[case]
    data = pd.read_csv(FIX / f"data_{case}.csv")
    items = pd.read_csv(FIX / f"items_{case}.csv")
    r = _results()
    return row["model"], data, items, r[r["case"] == case]


def _vector(r, quantity):
    return r[r["quantity"] == quantity]["value"].to_numpy(float)


def _matrix(r, quantity):
    v = _vector(r, quantity)
    n = int(round(np.sqrt(len(v))))
    return v.reshape(n, n) if len(v) else None


def _jacobian(items, model):
    """K = d(a1, d, logit g, logit u)/d(a, b, c, d) and second-derivative terms."""
    m = len(ACTIVE[model]); n = len(items) * m
    K = np.zeros((n, n)); second = []
    for j, (a, b, c, d) in enumerate(items[["a", "b", "c", "d"]].to_numpy(float)):
        o = j * m
        K[o, o] = 1.
        K[o + 1, o], K[o + 1, o + 1] = -b, -a
        K[o + 2, o + 2] = 1 / (c * (1 - c))
        second += [(o + 1, o, o + 1, -1.), (o + 2, o + 2, o + 2, -(1 - 2 * c) / (c * (1 - c)) ** 2)]
        if m == 4:
            K[o + 3, o + 3] = 1 / (d * (1 - d))
            second.append((o + 3, o + 3, o + 3, -(1 - 2 * d) / (d * (1 - d)) ** 2))
    return K, second


def _python_space_hessian(h_mirt, g_mirt, K, second, with_second_order=True):
    h = K.T @ h_mirt @ K
    if with_second_order:
        for k, i, j, val in second:
            h[i, j] += g_mirt[k] * val
            if i != j:
                h[j, i] += g_mirt[k] * val
    return h


def _info_on_grid(x, items, model, theta, logprior):
    """Observed information on an arbitrary grid: central differences of the
    module's analytic score (same construction as _observed_information)."""
    act = tuple(ACTIVE[model]); m = len(act); idx = ["abcd".index(q) for q in act]
    full = items[["a", "b", "c", "d"]].to_numpy(float)
    v = full[:, idx].reshape(-1); out = np.zeros((len(v), len(v)))
    for col in range(len(v)):
        h = 1e-4 * max(1., abs(v[col]))
        up, dn = full.copy(), full.copy()
        up[:, idx] = (v + h * (np.arange(len(v)) == col)).reshape(-1, m)
        dn[:, idx] = (v - h * (np.arange(len(v)) == col)).reshape(-1, m)
        out[:, col] = -(_analytic_marginal_score(x, up, act, theta, logprior)
                        - _analytic_marginal_score(x, dn, act, theta, logprior)) / (2 * h)
    return (out + out.T) / 2, _analytic_marginal_score(x, full, act, theta, logprior)


@pytest.mark.parametrize("case", FIXED)
def test_parameterization_matches_mirt_probtrace(case):
    model, _, items, r = _case(case)
    pt = r[r["quantity"] == "probtrace"]
    theta = pt["col"].astype(float).unique()
    expected = np.array([pt[pt["row"] == name]["value"].to_numpy(float)
                         for name in items["Madde"]]).T
    got = probabilities(theta, *(items[q].to_numpy(float) for q in "abcd"))
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("case", FIXED)
def test_marginal_loglik_matches_mirt_on_gauss_hermite_grid(case):
    _, data, items, r = _case(case)
    expected = _vector(r, "loglik_gh21")[0]
    got = marginal_loglik_at_fixed_parameters(data, items, points=POINTS)
    assert got == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("case", FIXED)
def test_marginal_score_matches_mirt_gradient(case):
    model, data, items, r = _case(case)
    K, _ = _jacobian(items, model)
    expected = K.T @ _vector(r, "gradient_internal_Richardson")
    got = marginal_score(data, items, model, points=POINTS)
    assert np.abs(got - expected).max() <= GRADIENT_RTOL * np.abs(got).max()


@pytest.mark.parametrize("case", FIXED)
def test_observed_information_matches_mirt_loglik_hessian(case):
    model, data, items, r = _case(case)
    K, second = _jacobian(items, model)
    g = _vector(r, "gradient_internal_Richardson")
    h_mirt = _matrix(r, "hessian_internal_Richardson")
    raw, reason = _observed_information(data.to_numpy(float), items, ACTIVE[model], POINTS, 4., .65)
    assert not reason
    info = (raw + raw.T) / 2
    scale = np.abs(info).max()
    reference = -_python_space_hessian(h_mirt, g, K, second)
    assert np.abs(info - reference).max() <= HESSIAN_RTOL * scale
    gap = np.linalg.norm(info - reference, 2)   # Weyl bound for eigenvalues
    assert np.abs(np.linalg.eigvalsh(info) - np.linalg.eigvalsh(reference)).max() <= gap * (1 + 1e-6) + 1e-9
    # The Jacobian's second-order term is not negligible away from the MLE.
    naive = -_python_space_hessian(h_mirt, g, K, second, with_second_order=False)
    assert np.abs(info - naive).max() > 100 * HESSIAN_RTOL * scale


@pytest.mark.parametrize("case", FIXED)
def test_standard_error_refusal_agrees_with_mirt(case):
    model, data, items, r = _case(case)
    _, reason = observed_information_se(data, items, model, points=POINTS)
    mirt_available = _matrix(r, "vcov_internal_default61_Richardson") is not None
    assert (not reason) == mirt_available
    if reason:
        assert "pozitif tanımlı değil" in reason


def test_standard_errors_match_mirt_gauss_hermite_hessian():
    model, data, items, r = _case("3pl_strong")
    K, second = _jacobian(items, model)
    info = -_python_space_hessian(_matrix(r, "hessian_internal_Richardson"),
                                  _vector(r, "gradient_internal_Richardson"), K, second)
    expected = np.sqrt(np.diag(np.linalg.inv(info))).reshape(len(items), -1)
    table, reason = observed_information_se(data, items, model, points=POINTS)
    assert not reason
    got = table[[f"SH({q})" for q in ACTIVE[model]]].to_numpy(float)
    np.testing.assert_allclose(got, expected, rtol=SE_RTOL, atol=0)


def test_mirt_se_type_on_default_grid():
    """mirt's own SE.type='Richardson' route, on its default 61-node grid."""
    model, data, items, r = _case("3pl_strong")
    x = data.to_numpy(float)
    theta = np.linspace(-6, 6, 61)
    w = np.exp(-theta ** 2 / 2); w /= w.sum()
    p = probabilities(theta, *(items[q].to_numpy(float) for q in "abcd"))
    ll = float(np.log(np.exp(x @ np.log(p).T + (1 - x) @ np.log1p(-p).T) @ w).sum())
    assert ll == pytest.approx(_vector(r, "loglik_default61")[0], rel=1e-12)
    info, score = _info_on_grid(x, items, model, theta, np.log(w))
    K, second = _jacobian(items, model)
    g = _vector(r, "gradient_internal_default61")
    assert np.abs(score - K.T @ g).max() <= GRADIENT_RTOL * np.abs(score).max()
    reference = -_python_space_hessian(-np.linalg.inv(_matrix(r, "vcov_internal_default61_Richardson")),
                                       g, K, second)
    assert np.abs(info - reference).max() <= HESSIAN_RTOL * np.abs(info).max()
    np.testing.assert_allclose(np.sqrt(np.diag(np.linalg.inv(info))),
                               np.sqrt(np.diag(np.linalg.inv(reference))), rtol=SE_RTOL)


def test_mirt_numerical_se_ignores_custom_prior_weights():
    """Documents mirt 1.48 behaviour behind the default-grid choice above."""
    model, data, items, r = _case("3pl_strong")
    x = data.to_numpy(float)
    gx, gw = hermgauss(POINTS)
    nodes = np.sqrt(2.) * gx
    dn = np.exp(-nodes ** 2 / 2); dn /= dn.sum()
    K, second = _jacobian(items, model)
    h_mirt = -np.linalg.inv(_matrix(r, "vcov_internal_customgh_Richardson"))
    info_dn, score_dn = _info_on_grid(x, items, model, nodes, np.log(dn))
    ref_dn = -_python_space_hessian(h_mirt, np.linalg.solve(K.T, score_dn), K, second)
    assert np.abs(info_dn - ref_dn).max() <= HESSIAN_RTOL * np.abs(info_dn).max()
    info_gh, score_gh = _info_on_grid(x, items, model, nodes, np.log(gw / np.sqrt(np.pi)))
    ref_gh = -_python_space_hessian(h_mirt, np.linalg.solve(K.T, score_gh), K, second)
    assert np.abs(info_gh - ref_gh).max() > 100 * HESSIAN_RTOL * np.abs(info_gh).max()


@lru_cache(maxsize=None)
def _python_free_fit():
    _, data, _, _ = _case("3pl_est")
    return fit_irt(data, "3PL", points=POINTS, max_iter=3000, tolerance=1e-8)


def test_free_3pl_estimates_match_mirt():
    model, data, _, r = _case("3pl_est")
    assert r[r["quantity"] == "prior_types"]["row"].iloc[0] == "none"
    assert _vector(r, "fit_converged")[0] == 1
    est = r[r["quantity"] == "fit_irtpars"].pivot(index="row", columns="col", values="value")
    est = est.loc[list(data.columns)]
    mirt_items = pd.DataFrame({"Madde": data.columns, "a": est["a"].to_numpy(float),
                               "b": est["b"].to_numpy(float), "c": est["g"].to_numpy(float), "d": 1.})
    mirt_ll = _vector(r, "fit_loglik")[0]
    # mirt's MLE is a stationary point of the Python likelihood (same objective).
    assert marginal_loglik_at_fixed_parameters(data, mirt_items, points=POINTS) == pytest.approx(mirt_ll, rel=1e-12)
    assert np.abs(marginal_score(data, mirt_items, model, points=POINTS)).max() / len(data) <= 1e-6
    fit = _python_free_fit()
    assert fit["converged"] and not fit["items"]["Sınır uyarısı"].any()
    # Python's EM stops slightly earlier (|dLL| < 1e-8 per cycle); differences
    # must be a negligible fraction of sampling error and never exceed mirt's maximum.
    assert -1e-6 <= mirt_ll - fit["loglik"] <= 1e-4
    se, reason = observed_information_se(data, mirt_items, model, points=POINTS)
    assert not reason
    for q in ACTIVE[model]:
        diff = np.abs(fit["items"][q].to_numpy(float) - mirt_items[q].to_numpy(float))
        assert (diff / se[f"SH({q})"].to_numpy(float)).max() <= .01


@pytest.mark.xfail(strict=True, reason="Open issue (B3): fit_irt does not reject structurally "
                   "unidentified models with more parameters than 2**k-1 response-pattern "
                   "degrees of freedom; mirt refuses 4PL with k=4 (16 > 15).")
def test_4pl_with_four_items_is_rejected_like_mirt():
    _, data, _, _ = _case("3pl_weak")    # any complete 0/1 data with k=4
    with pytest.raises(ValueError):
        fit_irt(data, "4PL", points=11, max_iter=1)
