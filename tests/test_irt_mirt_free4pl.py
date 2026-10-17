"""B5: free 4PL estimation compared with R mirt 1.48 (k=5 and k=8, N=3000).

References: tests/fixtures/mirt/mirt_free4pl_results.csv (scripts/mirt_free_4pl.R)
and python_free4pl_results.csv (scripts/python_free_4pl.py). Conditions are
matched: data, parameterization, N(0,1) latent fixed, the same 21-node
Gauss-Hermite grid, all four item parameters free, no item priors, the same
three starting values, tight stopping rules.

Findings encoded here (see docs/IRT_3PL_4PL_VALIDATION.md, B5):
* Fixed-parameter mathematics agrees at every mirt solution (logLik, gradient,
  observed information; derivatives compared in mirt's internal space a1, d,
  logit g, logit u, where the Jacobian does not amplify mirt's numerical error).
* Unbounded mirt solutions are not finite interior MLEs: slopes run off (a=22..166
  for one k=5 item depending on the start) along a flat ridge, the information
  matrix is numerically singular, and the points lie outside Python's box. They
  are not compared with Python estimates.
* Inside Python's box (a in [.15,4], c in [0,.35], d in [.65,1]) the mirt run
  with the same box and all Python starts lead, after direct L-BFGS-B maximization
  of the same likelihood, to one KKT point; both EMs stop slightly short of it.

Tolerances: logLik rel 1e-12 (observed <=2.6e-15); internal gradient abs 1e-6
(observed <=1.1e-7); internal information rel 1e-5 (observed <=1.8e-7). Values in
python_free4pl_results.csv are stored with 12 significant digits (logLik
resolution ~1e-7 at |logLik| ~1.5e4), hence STORED_ABS = 1e-6 when recomputing them.
"""
import os
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from core.irt import fit_irt
from core.irt_hessian import _analytic_marginal_score, marginal_score, observed_information_se
from core.irt_quadrature_validation import marginal_loglik_at_fixed_parameters
from core.irt_stationarity import stationarity_diagnostics

FIX = Path(__file__).parent / "fixtures" / "mirt"
MIRT = FIX / "mirt_free4pl_results.csv"
PY = FIX / "python_free4pl_results.csv"
if not (MIRT.exists() and PY.exists()):
    pytest.skip("Free 4PL reference values missing: run scripts/mirt_free_4pl.R and "
                "scripts/python_free_4pl.py (no values are invented).", allow_module_level=True)

SLOW = os.environ.get("PSIKOANALIZ_SLOW_TESTS") == "1"
STORED_ABS = 1e-6
CASES = list(pd.read_csv(FIX / "cases_free4pl.csv")["case"])
MIRT_RUNS = ["free_s0", "free_s-1", "free_s1", "box_s0"]
BOX = [(.15, 4.), (-4., 4.), (0., .35), (.65, 1.)]
GH_X, GH_W = hermgauss(21)
THETA, LOGPRIOR = np.sqrt(2.) * GH_X, np.log(GH_W / np.sqrt(np.pi))


@lru_cache(maxsize=None)
def _mirt():
    return pd.read_csv(MIRT, keep_default_na=False, na_values=["NA"])


@lru_cache(maxsize=None)
def _py():
    return pd.read_csv(PY)


@lru_cache(maxsize=None)
def _data(case):
    return pd.read_csv(FIX / f"data_{case}.csv")


def _mirt_value(case, run, quantity):
    r = _mirt()
    return r[(r.case == case) & (r.run == run) & (r.quantity == quantity)]


def _mirt_items(case, run):
    cols = list(_data(case).columns)
    e = _mirt_value(case, run, "estimate").pivot(index="row", columns="col", values="value").loc[cols]
    return pd.DataFrame({"Madde": cols, "a": e.a1.values, "b": -e.d.values / e.a1.values,
                         "c": e.g.values, "d": e.u.values})


def _py_items(case, source, run):
    p = _py()
    p = p[(p.case == case) & (p.source == source) & (p.run == run)]
    t = p.pivot(index="item", columns="param", values="value").loc[list(_data(case).columns)]
    return t.reset_index().rename(columns={"item": "Madde"})[["Madde", "a", "b", "c", "d"]], p.iloc[0]


def _internal(t):
    """mirt internal parameters (a1, d, logit g, logit u), item-major."""
    a, b, c, d = (t[q].to_numpy(float) for q in "abcd")
    return np.column_stack([a, -a * b, np.log(c / (1 - c)), np.log(d / (1 - d))]).reshape(-1)


def _internal_score(x, u):
    """Python analytic score mapped to mirt's internal space by the chain rule."""
    U = u.reshape(-1, 4)
    a, dm = U[:, 0], U[:, 1]
    c, d = 1 / (1 + np.exp(-U[:, 2])), 1 / (1 + np.exp(-U[:, 3]))
    full = np.column_stack([a, -dm / a, c, d])
    g = _analytic_marginal_score(x, full, ("a", "b", "c", "d"), THETA, LOGPRIOR).reshape(-1, 4)
    return np.column_stack([g[:, 0] + g[:, 1] * dm / a ** 2, -g[:, 1] / a,
                            g[:, 2] * c * (1 - c), g[:, 3] * d * (1 - d)]).reshape(-1)


def test_matched_setup_recorded_by_mirt():
    for case in CASES:
        assert _mirt_value(case, "setup", "prior_types")["row"].iloc[0] == "none"
        assert _mirt_value(case, "setup", "latent_estimated")["row"].iloc[0] == "MEAN_1 FALSE;COV_11 FALSE"
        for run in MIRT_RUNS:
            assert _mirt_value(case, run, "converged")["value"].iloc[0] == 1


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("run", MIRT_RUNS)
def test_loglik_equal_at_mirt_solution(case, run):
    expected = _mirt_value(case, run, "loglik")["value"].iloc[0]
    got = marginal_loglik_at_fixed_parameters(_data(case), _mirt_items(case, run), points=21)
    assert got == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("run", MIRT_RUNS)
def test_gradient_equal_in_mirt_internal_space(case, run):
    mirt_g = _mirt_value(case, run, "gradient_internal")["value"].to_numpy(float)
    got = _internal_score(_data(case).to_numpy(float), _internal(_mirt_items(case, run)))
    assert np.abs(got - mirt_g).max() <= 1e-6


@pytest.mark.parametrize("case", CASES)
def test_information_equal_and_singular_at_unbounded_mirt_solution(case):
    x = _data(case).to_numpy(float)
    u = _internal(_mirt_items(case, "free_s0"))
    n = len(u)
    hess = np.zeros((n, n))
    for i in range(n):
        h = 1e-5 * max(1., abs(u[i]))
        step = np.zeros(n); step[i] = h
        hess[:, i] = (_internal_score(x, u + step) - _internal_score(x, u - step)) / (2 * h)
    hess = (hess + hess.T) / 2
    mirt_h = _mirt_value(case, "free_s0", "hessian_internal_central")["value"].to_numpy(float).reshape(n, n)
    assert np.abs(hess - mirt_h).max() <= 1e-5 * np.abs(hess).max()
    for h in (hess, mirt_h):           # numerically singular: no reportable SE in either program
        eig = np.linalg.eigvalsh(-h)
        assert abs(eig[0]) <= 1e-6 * eig[-1]


@pytest.mark.parametrize("case", CASES)
def test_unbounded_mirt_solutions_are_not_finite_interior_mles(case):
    lls = [_mirt_value(case, r, "loglik")["value"].iloc[0] for r in MIRT_RUNS[:3]]
    box_ll = _mirt_value(case, "box_s0", "loglik")["value"].iloc[0]
    assert min(lls) > box_ll                                  # supremum lies outside Python's box
    slopes = np.array([_mirt_items(case, r)["a"].to_numpy() for r in MIRT_RUNS[:3]])
    assert (slopes.max(axis=1) > 4).all()                     # every start leaves the box via a slope
    runaway = slopes.max(axis=0).argmax()
    spread = np.ptp(slopes[:, runaway]) / slopes[:, runaway].min()
    assert spread > .1 and np.ptp(lls) < 2e-3                 # flat ridge: start-dependent slope, same logLik


@pytest.mark.parametrize("case", CASES)
def test_bounded_optimum_shared_by_mirt_and_python(case):
    x = _data(case)
    ref, meta = _py_items(case, "polished_from_mirt", "box_s0")
    assert marginal_loglik_at_fixed_parameters(x, ref, points=21) == pytest.approx(meta.loglik, abs=STORED_ABS)
    _, kkt = stationarity_diagnostics(x, ref, "4PL", points=21)
    assert kkt["max_KKT_violation"] <= 1e-6
    for run in ("s0", "s-1", "s1"):
        other, m = _py_items(case, "polished_from_python", run)
        assert abs(m.loglik - meta.loglik) <= 1e-5
        assert np.abs(other[list("abcd")].to_numpy() - ref[list("abcd")].to_numpy()).max() <= 1e-3


@pytest.mark.parametrize("case", CASES)
def test_both_ems_stop_close_to_bounded_optimum(case):
    _, optimum = _py_items(case, "polished_from_mirt", "box_s0")
    mirt_box = _mirt_value(case, "box_s0", "loglik")["value"].iloc[0]
    assert 0 <= optimum.loglik - mirt_box <= 5e-3
    for run in ("s0", "s-1", "s1"):
        em, m = _py_items(case, "python_em", run)
        assert marginal_loglik_at_fixed_parameters(_data(case), em, points=21) == pytest.approx(m.loglik, abs=STORED_ABS)
        assert 0 <= optimum.loglik - m.loglik <= 5e-3


@pytest.mark.parametrize("case", CASES)
def test_standard_errors_refused_at_bounded_solutions(case):
    x = _data(case)
    for source, run in (("polished_from_mirt", "box_s0"), ("python_em", "s0")):
        table, reason = observed_information_se(x, _py_items(case, source, run)[0], "4PL", points=21)
        assert table is None and "sınır" in reason


def _polish(x, table):
    k = len(table)
    def tab(v):
        t = pd.DataFrame(v.reshape(k, 4), columns=list("abcd")); t.insert(0, "Madde", table.Madde.values)
        return t
    lo, hi = np.array([b[0] for b in BOX] * k), np.array([b[1] for b in BOX] * k)
    v0 = np.clip(table[list("abcd")].to_numpy(float).reshape(-1), lo, hi)
    out = minimize(lambda v: -marginal_loglik_at_fixed_parameters(x, tab(v), points=21) / len(x), v0,
                   jac=lambda v: -marginal_score(x, tab(v), "4PL", points=21) / len(x),
                   method="L-BFGS-B", bounds=BOX * k, options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-11})
    return tab(out.x)


@pytest.mark.parametrize("case", [c for c in CASES if c.endswith("k8") or SLOW])
def test_live_polish_from_mirt_box_reproduces_stored_optimum(case):
    """Independent recomputation (k=8 ~2 s; k=5 ~13 s only with PSIKOANALIZ_SLOW_TESTS=1)."""
    x = _data(case)
    ref, meta = _py_items(case, "polished_from_mirt", "box_s0")
    got = _polish(x, _mirt_items(case, "box_s0"))
    assert marginal_loglik_at_fixed_parameters(x, got, points=21) == pytest.approx(meta.loglik, abs=1e-6)
    assert np.abs(got[list("abcd")].to_numpy() - ref[list("abcd")].to_numpy()).max() <= 1e-3


@pytest.mark.skipif(not SLOW, reason="~40 s refit; set PSIKOANALIZ_SLOW_TESTS=1")
def test_live_python_refit_matches_stored_k8():
    case = "4pl_free_k8"
    stored, meta = _py_items(case, "python_em", "s0")
    fit = fit_irt(_data(case), "4PL", points=21, max_iter=5000, tolerance=1e-9)
    assert fit["loglik"] == pytest.approx(meta.loglik, abs=STORED_ABS)
    assert np.abs(fit["items"][list("abcd")].to_numpy() - stored[list("abcd")].to_numpy()).max() <= 1e-9
