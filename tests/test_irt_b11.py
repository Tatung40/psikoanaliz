"""B11: 3PL/4PL local dependence, DIF and model fit.

Reference fixtures: tests/fixtures/irt_b11/r_b11_<model>.csv (scripts/mirt_irt_b11_reference.R;
R 4.3.3, mirt 1.48) from Python inputs (scripts/irt_b11_inputs_for_r.py).
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
import core.irt_mg34 as M
import core.irt_q3 as Q
from core.irt_mg import as_matrix, Quadrature, _parts

F = Path(__file__).parent / "fixtures" / "irt_b11"


@lru_cache(maxsize=None)
def _data(case):
    return pd.read_csv(F / f"data_{case}.csv", na_values="NA")


@lru_cache(maxsize=None)
def _ref(model):
    path = F / f"r_b11_{model}.csv"
    return pd.read_csv(path, keep_default_na=False, na_values=["NA"]) if path.exists() else None


def _r(model, case, quantity):
    r = _ref(model)
    return r[(r.case == case) & (r.quantity == quantity)]


def _xg(case):
    d = _data(case); x, obs = as_matrix(d.drop(columns="Grup"))
    return x, obs, (d["Grup"] == "F").to_numpy().astype(int), list(d.columns[1:])


def _mirt_items(model, case, cols, prefix="par"):
    p = _r(model, case, prefix)
    get = lambda nm, default: (p[p.col == nm].set_index("row").value.loc[cols].to_numpy()
                               if (p.col == nm).any() else np.full(len(cols), default))
    a, dm, c, d = get("a1", 1.), get("d", 0.), get("g", 0.), get("u", 1.)
    return a, -dm / a, c, d


# ------------------------------------------------------------------ IRF, constraints, design
def test_irf_parameterization_and_constraints():
    theta = np.array([-2., 0., 3.]); a = np.array([1.3, .8]); b = np.array([.2, -1.]); c = np.array([.2, .1])
    d = np.array([1., .9])
    P, L = M.irf(theta, a, b, c, d)
    np.testing.assert_allclose(P, c + (d - c) * expit(a * (theta[:, None] - b)))
    np.testing.assert_allclose(Q.probabilities(theta, a, b, c, d), P)
    np.testing.assert_allclose(Q.probabilities(theta, a, b), expit(a * (theta[:, None] - b)))   # B10 path unchanged
    three = M.Design34("3PL", ["anchor", "c", "all"], 2)
    assert (three.idx["d"] == -1).all() and three.idx["c"][0, 1] != three.idx["c"][1, 1]
    assert three.idx["a"][0, 1] == three.idx["a"][1, 1] and three.size == 3 + 4 + 6 + 2
    with pytest.raises(ValueError, match="tanımsız"):
        M.Design34("3PL", ["d", "anchor", "anchor"], 2)
    b4 = M.Design34("4PL", ["anchor"] * 3, 1).bounds()
    assert all(lo < hi for lo, hi in b4) and b4[2] == (0., .35) and b4[3] == (.65, 1.)            # 0 <= c < d <= 1
    with pytest.raises(ValueError, match="0 ≤ c < d ≤ 1"):
        Q.q3_analysis(_data("irt3pl_ok"), np.ones(12), np.zeros(12), "4PL", c=np.full(12, .4), d=np.full(12, .3))


@pytest.mark.parametrize("model,types", [("3PL", ["anchor", "b", "c", "a", "all"] + ["anchor"] * 7),
                                         ("4PL", ["anchor", "d", "c", "all"] + ["anchor"] * 8)])
@pytest.mark.parametrize("quadrature", ["gauss-hermite", "mirt"])
def test_multigroup_score_matches_finite_differences(model, types, quadrature):
    x, obs, g, _ = _xg("dif3pl")
    design = M.Design34(model, types, 2); parts = _parts(x, obs, g, 2); quad = Quadrature(quadrature, 31)
    p = M.start_values(x, obs, design) + np.random.default_rng(2).normal(0, .02, design.size)
    _, grad = M.negloglik(p, design, parts, quad, len(x))
    fd = np.array([(M.negloglik(p + e, design, parts, quad, len(x))[0]
                    - M.negloglik(p - e, design, parts, quad, len(x))[0]) / 2e-6 for e in np.eye(design.size) * 1e-6])
    assert np.abs(grad - fd).max() <= 1e-6 * np.abs(grad).max()


def test_bounded_fit_kkt_boundary_and_conditional_information():
    f = M.fit_single(_data("irt3pl_easy"), "3PL", points=31)
    assert f["converged"] and f["kkt_per_person"] <= 1e-6
    assert not f["regular"] and any(p == "c" for p, _, _ in f["bound_labels"])          # weakly identified c
    lo = np.array([b[0] for b in f["design"].bounds()]); hi = np.array([b[1] for b in f["design"].bounds()])
    on = np.where(f["at_bound"])[0]
    assert np.all(np.minimum(f["params"][on] - lo[on], hi[on] - f["params"][on]) <= 1e-3)
    assert f["cov_conditional"] and np.isnan(f["cov"][on[0], on[0]]) and np.isfinite(np.diag(f["cov"])[~f["at_bound"]]).all()


def test_nonconvergence_and_unidentified_linking():
    x, obs, g, _ = _xg("dif3pl_none")
    nc = M.fit(x, obs, g, M.Design34("3PL", ["anchor"] * 12, 2), max_iter=3)
    assert not nc["converged"] and nc["cov"] is None and "yakınsamadı" in nc["se_reason"]
    un = M.fit(x, obs, g, M.Design34("3PL", ["all"] * 12, 2), points=21)
    assert un["cov"] is None and not un["linked"] and "Çapa madde yok" in un["se_reason"]


# --------------------------------------------------------------------------- persons, Q3
def test_wle_general_reduces_to_2pl_and_handles_extremes():
    d = _data("irt3pl_ok"); x, obs = as_matrix(d)
    a = np.linspace(.8, 1.8, 12); b = np.linspace(-1, 1.5, 12)
    w2, _ = Q.wle(x, obs, a, b)
    wg, _, _, multi = Q.wle_general(x, obs, a, b, np.zeros(12), np.ones(12))
    assert not multi.any()
    np.testing.assert_allclose(wg, w2, atol=1e-10)
    xe = np.vstack([np.zeros(12), np.ones(12)]); oe = np.ones_like(xe, bool)
    t, _, edge, _ = Q.wle_general(xe, oe, a, b, np.full(12, .2), np.ones(12))
    assert edge.shape == (2,) and np.isfinite(t[1])


def test_q3_uses_the_full_3pl_irf_and_detects_planted_pair():
    """irt3pl_ld2: shared nuisance SD 2.5 on M5/M6 (with SD 1.2, irt3pl_ld, the fitted 3PL slopes absorb
    most of the dependence and the pair is not top-ranked -- documented sensitivity limit)."""
    d = _data("irt3pl_ld2"); f = M.fit_single(d, "3PL", points=31, information=False)
    a, b, c, dd = M.Design34("3PL", ["anchor"] * 12, 1).unpack(f["params"], 0)
    out = Q.q3_analysis(d, a, b, "3PL", person_method="EAP", c=c, points=31)
    assert {out["pairs"].iloc[0]["Madde 1"], out["pairs"].iloc[0]["Madde 2"]} == {"M5", "M6"}
    x, obs = as_matrix(d)
    two, _ = Q.q3_matrix(x, obs, out["theta"], a, b, 30)                                  # 2PL residuals differ
    assert np.nanmax(np.abs(two - out["q3"].to_numpy())) > 1e-3
    with pytest.raises(ValueError, match="c parametreleri"):
        Q.q3_analysis(d, a, b, "3PL")


# ----------------------------------------------------------------------------------- M2


# ------------------------------------------------------------------------- DIF utilities


# ---------------------------------------------------------------- R references: single group
SINGLE3 = ["irt3pl_ok", "irt3pl_ld", "irt3pl_missing", "irt3pl_easy"]


@lru_cache(maxsize=None)
def _box(model):
    path = F / f"r_b11_boxed_wle_{model}.csv"
    return pd.read_csv(path, keep_default_na=False, na_values=["NA"]) if path.exists() else None


def _b(model, case, quantity):
    r = _box(model)
    return r[(r.case == case) & (r.quantity == quantity)]


def _single_checks(model, case):
    d = _data(case); cols = list(d.columns); k = len(cols); x, obs = as_matrix(d)
    design = M.Design34(model, ["anchor"] * k, 1)
    pack = lambda a, b, c, dd: np.array([{"a": a, "b": b, "c": c, "d": dd}[p][i] for p, _, i in design.labels])
    # (1) likelihood at mirt's (unbounded) estimates: identical
    assert M.loglik_at(x, obs, np.zeros(len(x)), design, pack(*_mirt_items(model, case, cols))) == pytest.approx(
        float(_r(model, case, "loglik_fixed").value.iloc[0]), rel=1e-12)
    # (2) bounded estimates vs mirt restricted to the same box (lbound/ubound)
    p = _b(model, case, "box_par")
    get = lambda nm, default: (p[p.col == nm].set_index("row").value.loc[cols].to_numpy()
                               if (p.col == nm).any() else np.full(k, default))
    a, dm, c, dd = get("a1", 1.), get("d", 0.), get("g", 0.), get("u", 1.)
    f = M.fit_single(d, model, quadrature="mirt", information=False)
    A, B, C, DD = design.unpack(f["params"], 0)
    assert f["converged"] and f["loglik"] >= float(_b(model, case, "box_loglik").value.iloc[0]) - 1e-6
    tol = 5e-3 if model == "3PL" else 5e-2                     # 4PL: flat surface (logL gap < 4e-3)
    for ours, ref in ((A, a), (B, -dm / a), (C, c), (DD, dd)):
        assert np.abs(ours - ref).max() < tol                       # mirt's logit-scale g/u cannot reach the bound
    # (3) persons and Q3 with Python's parameters
    it = pd.read_csv(F / f"py_items_{case}.csv"); pe = pd.read_csv(F / f"py_persons_{case}.csv")
    grid = np.linspace(-6, 6, 61); lw = -grid ** 2 / 2; lw = lw - np.log(np.exp(lw).sum())
    e = Q.eap(x, obs, it.a, it.b, grid=(grid, lw), c=it.c, d=it.d)[0]
    assert np.abs(e - _r(model, case, "eap").value.to_numpy()).max() < 1e-6
    w, _, edge, multi = Q.wle_general(x, obs, it.a, it.b, it.c, it.d)
    cat = _b(model, case, "catR_wle").value.to_numpy()
    assert not edge.any() and np.abs(w - cat)[~multi].max() < 1e-4                        # Warm WLE, catR
    # non-unique Warm roots (3PL: 0-1 per case; 4PL: 54 and 207 of 3000) -> NaN; catR returns one of the roots
    assert np.isnan(w[multi]).all() and multi.sum() < (.002 if model == "3PL" else .1) * len(w)
    if model == "4PL":
        assert multi.any()
    q3, _ = Q.q3_matrix(x, obs, pe.eap.to_numpy(), it.a, it.b, 1, it.c, it.d)
    for _, r in _r(model, case, "q3").iterrows():
        assert q3[cols.index(r.row), cols.index(r.col)] == pytest.approx(r.value, abs=1e-12)
    return x, obs, it, d


@pytest.mark.parametrize("case", SINGLE3)
def test_3pl_single_group_against_mirt_and_catR(case):
    _single_checks("3PL", case)


def test_mirt_wle_is_the_jeffreys_mode_not_warm_for_3pl():
    """mirt fscores(WLE) maximizes L * sqrt(I): root of l' + I'/(2I), which differs from Warm's J/(2I)."""
    d = _data("irt3pl_ok"); x, obs = as_matrix(d.iloc[:40])
    it = pd.read_csv(F / "py_items_irt3pl_ok.csv")
    a, b, c, dd = (it[p].to_numpy() for p in "abcd")

    def info(t):
        P, L = M.irf(t, a, b, c, dd); d1 = (dd - c) * a * L * (1 - L)
        return (d1 ** 2 / (P * (1 - P))).sum(1)

    def score(t):
        P, L = M.irf(t, a, b, c, dd); d1 = (dd - c) * a * L * (1 - L)
        return ((x - P) * d1 / (P * (1 - P))).sum(1) + (info(t + 1e-5) - info(t - 1e-5)) / 2e-5 / (2 * info(t))
    lo, hi = np.full(len(x), -8.), np.full(len(x), 8.)
    for _ in range(80):
        mid = (lo + hi) / 2; pos = score(mid) > 0; lo = np.where(pos, mid, lo); hi = np.where(pos, hi, mid)
    mirt = _r("3PL", "irt3pl_ok", "wle").value.to_numpy()[:40]
    assert np.abs((lo + hi) / 2 - mirt).max() < 1e-4
    assert np.nanmax(np.abs(Q.wle_general(x, obs, a, b, c, dd)[0] - mirt)) > 1e-2


def test_information_matches_independent_numerical_hessian():
    """mg34 observed information == -Hessian of an independent GH marginal log-likelihood (interior truth)."""
    from numpy.polynomial.hermite_e import hermegauss
    d = _data("irt3pl_ok"); x, obs = as_matrix(d); k = d.shape[1]
    t = pd.read_csv(F / "truth_irt3pl_ok.csv")
    design = M.Design34("3PL", ["anchor"] * k, 1)
    par = np.array([t[p].to_numpy()[i] for p, _, i in design.labels])
    info = M.information(par, design, _parts(x, obs, np.zeros(len(x), int), 1), Quadrature("gauss-hermite", 31), len(x))
    nodes, weights = hermegauss(31); logw = np.log(weights / weights.sum())
    y = np.nan_to_num(np.asarray(x, float)); seen = np.asarray(obs, bool)

    def loglik(v):
        p = {name: np.zeros(k) for name in ("a", "b", "c")}
        for value, (name, _, item) in zip(v, design.labels):
            p[name][item] = value
        prob = p["c"] + (1 - p["c"]) * expit(p["a"] * (nodes[:, None] - p["b"]))
        terms = seen[:, None, :] * (y[:, None, :] * np.log(prob) + (1 - y[:, None, :]) * np.log1p(-prob))
        z = terms.sum(2) + logw
        top = z.max(1, keepdims=True)
        return float((top[:, 0] + np.log(np.exp(z - top).sum(1))).sum())

    h = 1e-4; n = len(par); ref = np.empty((n, n)); eye = np.eye(n) * h
    for i in range(n):
        for j in range(i, n):
            ref[i, j] = ref[j, i] = -(loglik(par + eye[i] + eye[j]) - loglik(par + eye[i] - eye[j])
                                      - loglik(par - eye[i] + eye[j]) + loglik(par - eye[i] - eye[j])) / (4 * h * h)
    np.testing.assert_allclose(info, ref, rtol=1e-4, atol=1e-5 * np.abs(ref).max())


# ------------------------------------------------------------------- R references: DIF (3PL)


def _fake(converged=True, regular=True, linked=True, bound=(), ll=-100.):
    return {"converged": converged, "regular": regular, "linked": linked, "loglik": ll, "bound_labels": list(bound),
            "cov": np.eye(1), "cov_conditional": False, "quadrature_ok": True}


# ------------------------------------------------------------------- Phase B: 4PL references
@pytest.mark.parametrize("case", ["irt4pl_ok", "irt4pl_ld"])
def test_4pl_single_group_against_mirt_and_catR(case):
    _single_checks("4PL", case)

