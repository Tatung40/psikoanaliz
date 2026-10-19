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
import core.irt_globalfit as GF
import core.irt_dif34 as D
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
@pytest.mark.parametrize("model", ["3PL", "4PL"])
@pytest.mark.parametrize("grid", ["gauss-hermite", "mirt"])
def test_m2_jacobian_df_and_fixed_columns(model, grid):
    case = "irt3pl_ok" if model == "3PL" else "irt4pl_ok"
    x = _data(case).to_numpy(float); n, k = x.shape
    rng = np.random.default_rng(3)
    a, b = rng.uniform(.8, 2, k), np.linspace(-.4, 2, k); c = rng.uniform(.1, .25, k)
    d = rng.uniform(.88, .98, k) if model == "4PL" else np.ones(k)
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    theta, w = GF.quadrature(grid, 41)
    names = GF.FREE[model]

    def margins(par):
        v = dict(zip(names, np.split(par, len(names))))
        P = v.get("c", c) + (v.get("d", d) - v.get("c", c)) * expit(v["a"] * (theta[:, None] - v["b"]))
        return np.r_[w @ P, [w @ (P[:, i] * P[:, j]) for i, j in pairs]]

    par = np.concatenate([{"a": a, "b": b, "c": c, "d": d}[nm] for nm in names])
    jac = np.column_stack([(margins(par + e) - margins(par - e)) / 2e-6 for e in np.eye(len(par)) * 1e-6])
    ours = GF.m2_statistic(x, a, b, model, grid=grid, c=c, d=d, max_condition=np.inf)
    np.testing.assert_allclose(ours["jacobian"], jac, atol=1e-8)
    s = k + k * (k - 1) // 2
    assert ours["df"] == s - len(names) * k and not ours["conditional"]
    fixed = {"c": np.r_[True, True, np.zeros(k - 2, bool)]}
    cond = GF.m2_statistic(x, a, b, model, grid=grid, c=c, d=d, fixed=fixed, max_condition=np.inf)
    assert cond["conditional"] and cond["df"] == ours["df"] + 2 and cond["free_parameters"] == ours["free_parameters"] - 2
    np.testing.assert_allclose(cond["jacobian"], np.delete(jac, [2 * k, 2 * k + 1], axis=1), atol=1e-8)


def test_m2_unavailable_df_and_boundary_policy():
    x = _data("irt3pl_ok").iloc[:, :5]                                                    # s = 15 < q = 15 + 1
    assert "Serbestlik" in GF.m2_statistic(x, np.ones(5), np.zeros(5), "3PL", c=np.full(5, .2))["reason"]
    d = _data("irt3pl_easy"); k = d.shape[1]
    f = M.fit_single(d, "3PL", points=31, information=False)
    a, b, c, _ = M.Design34("3PL", ["anchor"] * k, 1).unpack(f["params"], 0)
    fixed = {p_: np.zeros(k, bool) for p_ in "abc"}
    for p_, _, i in f["bound_labels"]:
        fixed[p_][i] = True
    g = GF.global_fit_34(d, a, b, c, None, "3PL", np.zeros(len(d)), "EAP", True, fixed, points=31)
    assert g["m2"]["available"] and g["m2"]["conditional"] and np.isnan(g["m2"]["p_reported"])
    assert g["classification"].iloc[0]["Sınıf"] == GF.CLASS_EXPERIMENTAL
    miss = GF.global_fit_34(_data("irt3pl_missing"), a, b, c, None, "3PL", np.zeros(2000), "EAP", True, None, 31)
    assert not miss["m2"]["available"] and "eksiksiz" in miss["m2"]["reason"]


# ------------------------------------------------------------------------- DIF utilities
def test_response_function_comparison_shapes():
    s = D.response_function_comparison([1.5, 0., .2, 1.], [1.5, .5, .2, 1.], 0., 1.)
    assert s["shape"] == "against" and s["signed"] < 0 and s["unsigned"] == pytest.approx(-s["signed"], rel=1e-9)
    x = D.response_function_comparison([1.8, 0., .2, 1.], [.7, 0., .2, 1.], 0., 1.)
    assert x["shape"] == "crossing" and any(abs(v) < .05 for v in x["crossings"])
    lower = D.response_function_comparison([1.5, 0., .1, 1.], [1.5, 0., .3, 1.], 0., 1.)
    assert lower["shape"] == "favour" and lower["max_abs"] > .15                             # c-DIF: low-θ region
    assert D.response_function_comparison([1., 0., .2, 1.], [1., 0., .2, 1.], 0., 1.)["shape"] == "none"


def test_dif34_guards():
    d = _data("dif3pl_none")
    with pytest.raises(ValueError, match="en az 100"):
        small = pd.concat([d[d.Grup == "R"].iloc[:300], d[d.Grup == "F"].iloc[:60]])
        D.dif_analysis_34(small.drop(columns="Grup"), small["Grup"], "R", "F")
    with pytest.raises(ValueError, match="yalnızca 3PL ve 4PL"):
        D.dif_analysis_34(d.drop(columns="Grup"), d["Grup"], "R", "F", model="2PL")
    dis = d.copy(); items = list(dis.columns[1:])
    dis.loc[dis.Grup == "R", items[:6]] = np.nan; dis.loc[dis.Grup == "F", items[6:]] = np.nan
    with pytest.raises(ValueError, match="Bağlantısız"):
        D.dif_analysis_34(dis.drop(columns="Grup"), dis["Grup"], "R", "F")


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


def _mirt_style_m2(x, a, b, c, d):
    """M2 with mirt 1.48's dichotomous dP (theta*P*Q, P*Q, 1-P): the 2PL derivatives."""
    n, k = x.shape; theta, w = GF.quadrature("mirt", 41)
    P = c + (d - c) * expit(a * (theta[:, None] - b))
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]; sets = [(i,) for i in range(k)] + pairs; s = len(sets)
    ind = np.zeros((s, k))
    for r, S in enumerate(sets):
        ind[r, list(S)] = 1
    m = np.exp(ind @ np.log(P).T); pi = m @ w
    union = np.maximum(ind[:, None, :], ind[None, :, :]).reshape(s * s, k)
    xi = (np.exp(union @ np.log(P).T) @ w).reshape(s, s) - np.outer(pi, pi)
    e = np.r_[x.mean(0), [np.mean(x[:, i] * x[:, j]) for i, j in pairs]] - pi
    cols = []
    for j in range(k):
        mP = m * ind[:, [j]] / P[:, j][None, :]
        parts = [theta * P[:, j] * (1 - P[:, j]), P[:, j] * (1 - P[:, j]), 1 - P[:, j]] + ([P[:, j]] if np.any(d < 1) else [])
        cols.append(np.column_stack([(mP * v[None, :]) @ w for v in parts]))
    delta = np.hstack(cols); Wi = np.linalg.solve(xi, e); Wd = np.linalg.solve(xi, delta)
    return n * (e @ Wi - (delta.T @ Wi) @ np.linalg.solve(delta.T @ Wd, delta.T @ Wi))


@pytest.mark.parametrize("case", ["irt3pl_ok", "irt3pl_ld", "irt3pl_easy"])
def test_m2_differs_from_mirt_only_through_mirts_2pl_jacobian(case):
    model = "3PL"; d = _data(case); cols = list(d.columns); a, b, c, dd = _mirt_items(model, case, cols)
    ref = _r(model, case, "m2").set_index("col").value
    x = d.to_numpy(float)
    assert _mirt_style_m2(x, a, b, c, dd) == pytest.approx(ref["M2"], rel=1e-9)               # rest identical
    ours = GF.m2_statistic(d, a, b, model, grid="mirt", c=c, d=dd, max_condition=np.inf)
    assert ours["df"] == int(ref["df"]) and ours["SRMSR"] == pytest.approx(ref["SRMSR"], rel=1e-9)
    assert abs(ours["M2"] - ref["M2"]) > .1                                                   # correct Jacobian differs


def test_information_matches_b2_hessian_validated_against_mirt():
    """mg34 observed information == core.irt_hessian (B2/B3, checked against mirt) at interior parameters."""
    from core.irt_hessian import _observed_information
    d = _data("irt3pl_ok"); x, obs = as_matrix(d); k = d.shape[1]
    t = pd.read_csv(F / "truth_irt3pl_ok.csv")
    design = M.Design34("3PL", ["anchor"] * k, 1)
    par = np.array([t[p].to_numpy()[i] for p, _, i in design.labels])
    info = M.information(par, design, _parts(x, obs, np.zeros(len(x), int), 1), Quadrature("gauss-hermite", 31), len(x))
    ref, why = _observed_information(x, t, ("a", "b", "c"), 31, 4., .65)
    assert ref is not None, why
    np.testing.assert_allclose(info, (ref + ref.T) / 2, rtol=1e-4, atol=1e-5 * np.abs(ref).max())


# ------------------------------------------------------------------- R references: DIF (3PL)
@pytest.mark.parametrize("case", ["dif3pl", "dif3pl_none"])
def test_3pl_round0_lr_and_final_model_match_boxed_mirt(case):
    x, obs, g, cols = _xg(case); k = len(cols)
    fitter = D._Fitter34(x, obs, g, "3PL", "mirt", 41, M.default_bounds())
    base = fitter(["anchor"] * k)
    assert base["loglik"] >= float(_b("3PL", case, "box_base_loglik").value.iloc[0]) - 1e-6
    ref = _b("3PL", case, "box_lr_omnibus").set_index("row").value
    for i, c in enumerate(cols):
        t = ["anchor"] * k; t[i] = "all"
        stat, _, _ = D._lr(fitter(t), base, i)
        assert stat == pytest.approx(ref[c], abs=5e-3)                  # mirt EM tolerance at the box
    anchors = pd.read_csv(F / f"py_dif_anchors_{case}.csv").anchor.tolist()
    fin = fitter(["anchor" if c in anchors else "all" for c in cols])
    assert fin["converged"] and fin["loglik"] >= float(_b("3PL", case, "box_final_loglik").value.iloc[0]) - 1e-6
    # unbounded mirt: same likelihood function at mirt's parameters (fixed-parameter evaluation)
    pR, pF = _r("3PL", case, "final_par_R"), _r("3PL", case, "final_par_F")
    design = fin["design"]
    get = lambda p, nm: p[p.col == nm].set_index("row").value.loc[cols].to_numpy()
    vals = {0: {"a": get(pR, "a1"), "b": -get(pR, "d") / get(pR, "a1"), "c": get(pR, "g")},
            1: {"a": get(pF, "a1"), "b": -get(pF, "d") / get(pF, "a1"), "c": get(pF, "g")}}
    par = np.array([vals[max(gg, 0)][p][i] for p, gg, i in design.labels]
                   + [float(pF[pF.col == "MEAN_1"].value.iloc[0]), .5 * np.log(float(pF[pF.col == "COV_11"].value.iloc[0]))])
    assert M.loglik_at(x, obs, g, design, par) == pytest.approx(float(_r("3PL", case, "final_loglik_fixed_sum").value.iloc[0]),
                                                                 rel=1e-12)


def test_3pl_dif_procedure_on_planted_and_null_data():
    known = D.dif_analysis_34(_data("dif3pl").drop(columns="Grup"), _data("dif3pl")["Grup"], "R", "F", points=31)
    t = known["table"].set_index("Madde")
    removed = set(t.index[~t["Çapa"]])
    assert {"M3", "M8", "M10"} <= removed                          # all planted DIF items leave the anchor set
    assert (t.loc[["M3", "M8", "M10"], "LR χ² genel (sd=3)"] > 25).all()
    for item in removed:                                            # boundary items: no p-value, only screening
        if not np.isfinite(t.loc[item, "p"]):
            assert t.loc[item, "Çapa taraması"] == "Çapa dışı (sezgisel)" and not t.loc[item, "DIF işareti"]
    assert t.loc["M3", "İşaretli fark E_F[P_F − P_R]"] < 0          # b +0.6: harder for focal
    assert t.loc["M8", "İşaretli fark E_F[P_F − P_R]"] > 0          # c +0.15: easier at low theta
    ok = t[t["Çıkarım"] == "Asimptotik χ² (iç nokta)"]
    assert len(ok) and np.isfinite(ok["p"]).all()


# ------------------------------------------------------------------- Phase B: 4PL references
@pytest.mark.parametrize("case", ["irt4pl_ok", "irt4pl_ld"])
def test_4pl_single_group_against_mirt_and_catR(case):
    _single_checks("4PL", case)


@pytest.mark.parametrize("case", ["irt4pl_ok", "irt4pl_ld"])
def test_4pl_m2_jacobian_vs_mirt_and_policy(case):
    d = _data(case); cols = list(d.columns); a, b, c, dd = _mirt_items("4PL", case, cols)
    ref = _r("4PL", case, "m2").set_index("col").value
    assert _mirt_style_m2(d.to_numpy(float), a, b, c, dd) == pytest.approx(ref["M2"], rel=1e-8)
    ours = GF.m2_statistic(d, a, b, "4PL", grid="mirt", c=c, d=dd, max_condition=np.inf)
    assert ours["df"] == int(ref["df"]) == 78 - 48
    f = M.fit_single(d, "4PL", points=31, information=False)                       # production-like bounded fit
    A, B, C, DD = M.Design34("4PL", ["anchor"] * 12, 1).unpack(f["params"], 0)
    fixed = {p: np.zeros(12, bool) for p in "abcd"}
    for p, _, i in f["bound_labels"]:
        fixed[p][i] = True
    g = GF.global_fit_34(d, A, B, C, DD, "4PL", np.zeros(len(d)), "EAP", f["converged"], fixed, points=31)
    assert f["bound_labels"] and g["classification"].iloc[0]["Sınıf"] in (GF.CLASS_EXPERIMENTAL, "Hesaplanmadı")
    assert not np.isfinite(g["m2"].get("p_reported", np.nan))                       # never inferential at a bound


@pytest.mark.parametrize("case", ["dif4pl", "dif4pl_none"])
def test_4pl_descriptive_lr_matches_boxed_mirt(case):
    x, obs, g, cols = _xg(case); k = len(cols)
    fitter = D._Fitter34(x, obs, g, "4PL", "mirt", 41, M.default_bounds())
    base = fitter(["anchor"] * k)
    assert base["loglik"] >= float(_b("4PL", case, "box_base_loglik").value.iloc[0]) - 1e-6
    ref = _b("4PL", case, "box_lr_omnibus").set_index("row").value
    diffs = []
    for i, c in enumerate(cols):
        t = ["anchor"] * k; t[i] = "all"
        diffs.append(D._lr(fitter(t), base, i)[0] - ref[c])
    assert np.max(np.abs(diffs)) < 1e-2                    # flat 4PL surfaces; mirt TOL 1e-6 at the box
    assert np.min(diffs) > -5e-3                           # ours never materially below mirt
    anchors = pd.read_csv(F / f"py_dif_anchors_{case}.csv").anchor.tolist()
    design = M.Design34("4PL", ["anchor" if c in anchors else "all" for c in cols], 2)
    pR, pF = _r("4PL", case, "final_par_R"), _r("4PL", case, "final_par_F")
    get = lambda p, nm: p[p.col == nm].set_index("row").value.loc[cols].to_numpy()
    vals = {g_: {"a": get(p_, "a1"), "b": -get(p_, "d") / get(p_, "a1"), "c": get(p_, "g"), "d": get(p_, "u")}
            for g_, p_ in ((0, pR), (1, pF))}
    par = np.array([vals[max(gg, 0)][p][i] for p, gg, i in design.labels]
                   + [float(pF[pF.col == "MEAN_1"].value.iloc[0]), .5 * np.log(float(pF[pF.col == "COV_11"].value.iloc[0]))])
    assert M.loglik_at(x, obs, g, design, par) == pytest.approx(float(_r("4PL", case, "final_loglik_fixed_sum").value.iloc[0]),
                                                                 rel=1e-12)


def test_4pl_dif_descriptive_recovers_location_dif():
    d = _data("dif4pl")
    out = D.dif_analysis_34(d.drop(columns="Grup"), d["Grup"], "R", "F", model="4PL", points=31, inferential=False)
    t = out["table"].set_index("Madde")
    assert t["p"].isna().all() and not out["purify"] and set(t["Çıkarım"]) == {"Betimsel"}
    assert t.loc["M3", "LR χ² genel (sd=4)"] == t["LR χ² genel (sd=4)"].max()
    assert t.loc["M3", "İşaretli fark E_F[P_F − P_R]"] < -.05


def test_weakly_identified_c_small_sample_is_non_regular():
    f = M.fit_single(_data("irt3pl_weak"), "3PL", points=21)
    assert f["converged"] and not f["regular"] and len(f["bound_labels"]) >= 3
    x = _data("irt3pl_weak")
    a, b, c, _ = M.Design34("3PL", ["anchor"] * 8, 1).unpack(f["params"], 0)
    fixed = {p_: np.zeros(8, bool) for p_ in "abc"}
    for p_, _, i in f["bound_labels"]:
        fixed[p_][i] = True
    g = GF.global_fit_34(x, a, b, c, None, "3PL", np.zeros(len(x)), "EAP", True, fixed, points=21)
    assert g["classification"].iloc[0]["Sınıf"] != GF.CLASS_INFERENTIAL
