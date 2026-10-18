"""B10: 1PL/2PL local dependence, DIF and model fit.

Reference fixtures: tests/fixtures/irt_b10/r_b10_results.csv (scripts/mirt_irt_b10_reference.R;
R 4.3.3, mirt 1.48) from Python inputs (scripts/irt_b10_inputs_for_r.py).
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
import core.irt_mg as M
import core.irt_q3 as Q
import core.irt_dif as D
import core.irt_globalfit as GF
from core.irt import fit_irt

F = Path(__file__).parent / "fixtures" / "irt_b10"


@lru_cache(maxsize=None)
def _data(case):
    return pd.read_csv(F / f"data_{case}.csv", na_values="NA")


@lru_cache(maxsize=None)
def _ref():
    return pd.read_csv(F / "r_b10_results.csv", keep_default_na=False, na_values=["NA"])


def _r(case, quantity):
    r = _ref()
    return r[(r.case == case) & (r.quantity == quantity)]


def _xg(case):
    d = _data(case); x, obs = M.as_matrix(d.drop(columns="Grup"))
    return x, obs, (d["Grup"] == "F").to_numpy().astype(int), list(d.columns[1:])


@lru_cache(maxsize=None)
def _dif(case, model):
    d = _data(case)
    return D.dif_analysis(d.drop(columns="Grup"), d["Grup"], "R", "F", model=model)


# ------------------------------------------------------------- response functions / design
def test_response_function_and_parameterization():
    theta = np.array([-1., 0., 2.]); a = np.array([1., 1.7]); b = np.array([.5, -.3])
    P = Q.probabilities(theta, a, b)
    np.testing.assert_allclose(P, expit(a * (theta[:, None] - b)))
    np.testing.assert_allclose(np.log(P / (1 - P)), a * theta[:, None] + (-a * b))        # d = -a b
    one = M.Design("1PL", ["anchor", "b", "anchor"], 2)
    assert (one.a == -1).all() and one.size == 4 + 2                                       # a fixed at 1
    two = M.Design("2PL", ["anchor", "b", "ab"], 2)
    assert two.a[0, 1] == two.a[1, 1] and two.a[0, 2] != two.a[1, 2] and two.b[0, 1] != two.b[1, 1]
    with pytest.raises(ValueError, match="1PL"):
        M.Design("1PL", ["ab", "anchor", "anchor"], 2)


@pytest.mark.parametrize("model,types", [("1PL", ["anchor", "b"] + ["anchor"] * 10),
                                         ("2PL", ["anchor", "b", "ab"] + ["anchor"] * 9)])
@pytest.mark.parametrize("quadrature", ["gauss-hermite", "mirt"])
def test_multigroup_score_matches_finite_differences(model, types, quadrature):
    x, obs, g, _ = _xg("dif1pl_uniform")
    design = M.Design(model, types, 2); parts = M._parts(x, obs, g, 2); quad = M.Quadrature(quadrature, 41)
    p = M.start_values(x, obs, design, g) + np.random.default_rng(1).normal(0, .1, design.size)
    _, grad = M.negloglik(p, design, parts, quad, len(x))
    fd = np.array([(M.negloglik(p + e, design, parts, quad, len(x))[0]
                    - M.negloglik(p - e, design, parts, quad, len(x))[0]) / 2e-6 for e in np.eye(design.size) * 1e-6])
    assert np.abs(grad - fd).max() <= 1e-6 * np.abs(grad).max()


@pytest.mark.parametrize("model", ["1PL", "2PL"])
def test_single_group_mml_agrees_with_production_em(model):
    d = _data("irt1pl_ok" if model == "1PL" else "irt2pl_ok")
    ours = M.fit_single(d, model, points=31)
    em = fit_irt(d, model=model, points=31, max_iter=2000, tolerance=1e-9)
    a, b = M.Design(model, ["anchor"] * 10, 1).unpack(ours["params"], 0)
    assert ours["converged"] and np.abs(b - em["items"]["b"]).max() < 1e-4
    assert np.abs(a - em["items"]["a"]).max() < 1e-4
    assert ours["loglik"] == pytest.approx(em["loglik"], abs=1e-6)


def test_unidentified_linking_gives_no_standard_errors():
    """All items group-specific: no anchors, the focal metric is not identified."""
    x, obs, g, _ = _xg("dif2pl_none")
    f = M.fit(x, obs, g, M.Design("2PL", ["ab"] * 12, 2))
    assert f["cov"] is None and f["se_reason"]


def test_nonconvergence_withholds_inference(monkeypatch):
    x, obs, g, _ = _xg("dif2pl_none")
    f = M.fit(x, obs, g, M.Design("2PL", ["anchor"] * 12, 2), max_iter=2)
    assert not f["converged"] and f["cov"] is None and "yakınsamadı" in f["se_reason"]
    original = D.mg_fit
    monkeypatch.setattr(D, "mg_fit", lambda *a, **k: original(*a, **dict(k, max_iter=2)))
    d = _data("dif2pl_none")
    out = D.dif_analysis(d.drop(columns="Grup"), d["Grup"], "R", "F", model="2PL", purify=False)
    assert out["table"]["p"].isna().all() and not out["table"]["DIF işareti"].any()
    assert out["table"]["Not"].str.contains("p değeri verilmedi").all()


# ------------------------------------------------------------------------ DIF procedure
def test_1pl_uniform_dif_detected_and_sign():
    out = _dif("dif1pl_uniform", "1PL")
    t = out["table"].set_index("Madde")
    assert set(t.index[t["DIF işareti"]]) == {"M3", "M9"} and out["stable"]
    assert t.loc["M3", "Δb (b_odak − b_ref)"] > 0 > t.loc["M9", "Δb (b_odak − b_ref)"]
    assert t.loc["M3", "İşaretli fark E_F[P_F − P_R]"] < 0                       # harder for focal
    assert "Δa (a_odak − a_ref)" not in t and (t.loc[["M3", "M9"], "DIF türü"] == D.DIF_TYPES["uniform"]).all()
    np.testing.assert_allclose(t["Wald z"], t["Δb (b_odak − b_ref)"] / t["SH(Δb)"])
    assert out["joint"]["mu_F"] < -.2                                                      # impact absorbed


def test_2pl_uniform_and_nonuniform_dif_classified_from_response_functions():
    out = _dif("dif2pl_mixed", "2PL")
    t = out["table"].set_index("Madde")
    assert set(t.index[t["DIF işareti"]]) == {"M3", "M7"}
    assert t.loc["M3", "DIF türü"] == D.DIF_TYPES["uniform"] and t.loc["M3", "p (tek biçimli olmayan)"] > .05
    assert t.loc["M7", "DIF türü"] == D.DIF_TYPES["crossing"] and t.loc["M7", "Δa (a_odak − a_ref)"] < -.5
    assert np.isfinite(t.loc["M7", "Kesişim θ*"]) and abs(t.loc["M7", "Kesişim θ*"]) < 1
    # Non-uniform: the signed area understates the unsigned one; uniform: they agree.
    assert t.loc["M7", "İşaretsiz fark E_F|P_F − P_R|"] > 1.5 * abs(t.loc["M7", "İşaretli fark E_F[P_F − P_R]"])
    assert t.loc["M3", "İşaretsiz fark E_F|P_F − P_R|"] == pytest.approx(-t.loc["M3", "İşaretli fark E_F[P_F − P_R]"], rel=1e-6)
    # LR decomposition: omnibus = uniform + non-uniform
    np.testing.assert_allclose(t["LR χ² genel (sd=2)"], t["LR χ² tek biçimli (sd=1)"] + t["LR χ² tek biçimli olmayan (sd=1)"],
                               atol=1e-6)


def test_2pl_no_dif_control_and_purification():
    none = _dif("dif2pl_none", "2PL")
    assert not none["table"]["DIF işareti"].any() and len(none["anchors"]) == 12
    mixed = _dif("dif2pl_mixed", "2PL")
    assert "M3" not in mixed["anchors"] and "M7" not in mixed["anchors"]
    assert list(mixed["history"]["Tur"]) == [0, 1] and mixed["history"]["İşaretlenen"].iloc[0] == ["M3", "M7"]


def test_response_function_summary_crossing_point():
    s = D.response_function_summary(1.5, .2, .5, .2, 0., 1.)
    assert s["crossing"] == pytest.approx(.2) and s["crossing_inside"]
    u = D.response_function_summary(1., 0., 1., .5, 0., 1.)
    assert np.isnan(u["crossing"]) and u["signed"] < 0 and u["unsigned"] == pytest.approx(-u["signed"])


def test_dif_guards_small_unbalanced_untestable_and_disconnected():
    d = _data("dif2pl_none")
    small = pd.concat([d[d.Grup == "R"].iloc[:400], d[d.Grup == "F"].iloc[:60]])
    out = D.dif_analysis(small.drop(columns="Grup"), small["Grup"], "R", "F", model="1PL", purify=False)
    assert any("küçük" in w for w in out["warnings"]) and any("dengesiz" in w for w in out["warnings"])
    with pytest.raises(ValueError, match="en az 50"):
        tiny = pd.concat([d[d.Grup == "R"].iloc[:200], d[d.Grup == "F"].iloc[:30]])
        D.dif_analysis(tiny.drop(columns="Grup"), tiny["Grup"], "R", "F")
    with pytest.raises(ValueError, match="yalnızca 1PL ve 2PL"):
        D.dif_analysis(d.drop(columns="Grup"), d["Grup"], "R", "F", model="3PL")
    with pytest.raises(ValueError, match="farklı"):
        D.dif_analysis(d.drop(columns="Grup"), d["Grup"], "R", "R")
    c = d.copy(); c.loc[c.Grup == "F", "M1"] = 1.
    out = D.dif_analysis(c.drop(columns="Grup"), c["Grup"], "R", "F", model="1PL", purify=False)
    row = out["table"].set_index("Madde").loc["M1"]
    assert "aynı" in row["Not"] and not row["DIF işareti"] and np.isnan(row.get("p", np.nan))
    dis = d.copy(); items = list(dis.columns[1:])
    dis.loc[dis.Grup == "R", items[:6]] = np.nan; dis.loc[dis.Grup == "F", items[6:]] = np.nan
    with pytest.raises(ValueError, match="Bağlantısız"):
        D.dif_analysis(dis.drop(columns="Grup"), dis["Grup"], "R", "F")


# ----------------------------------------------------------------------------------- Q3
def test_q3_identity_adjustment_and_planted_pair():
    d = _data("irt2pl_ld"); r = fit_irt(d, model="2PL", points=41, max_iter=500)
    out = Q.q3_analysis(d, r["items"]["a"], r["items"]["b"], "2PL")
    assert {out["pairs"].iloc[0]["Madde 1"], out["pairs"].iloc[0]["Madde 2"]} == {"M4", "M5"}
    off = out["q3"].to_numpy()[np.triu_indices(10, 1)]
    np.testing.assert_allclose(out["adjusted"].to_numpy()[np.triu_indices(10, 1)], off - off.mean())
    assert out["expected_under_li"] == pytest.approx(-1 / 9)


def test_q3_missing_pairwise_zero_variance_and_guards():
    x = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 0, 1]], float); obs = np.ones_like(x, bool); obs[0, 1] = False
    q3, counts = Q.q3_matrix(x, obs, np.zeros(4), np.ones(3), np.array([0., 0., -60.]), min_pairs=1)
    assert counts[0, 1] == 3 and np.isfinite(q3[0, 1]) and np.isnan(q3[0, 2])
    d = _data("irt2pl_ok")
    with pytest.raises(ValueError, match="1PL için"):
        Q.q3_analysis(d, np.full(10, 1.3), np.zeros(10), "1PL")
    with pytest.raises(ValueError, match="Bootstrap"):
        Q.q3_analysis(d, np.ones(10), np.zeros(10), "2PL", bootstrap=10_000)
    with pytest.raises(ValueError, match="WLE veya EAP"):
        Q.q3_analysis(d, np.ones(10), np.zeros(10), "2PL", person_method="ML")


def test_q3_bootstrap_reports_failures_and_is_reproducible():
    d = _data("irt1pl_ok"); r = fit_irt(d, model="1PL", points=31, max_iter=500)
    calls = {"n": 0}

    def refit(frame):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("simulated failure")
        f = fit_irt(frame, model="1PL", points=31, max_iter=500)
        return f["items"]["a"].to_numpy(), f["items"]["b"].to_numpy(), f["converged"]

    out = Q.q3_analysis(d, r["items"]["a"], r["items"]["b"], "1PL", bootstrap=6, seed=4, refit=refit, points=31)
    bs = out["bootstrap"]
    assert bs["failed"] == 1 and bs["used"] == 5 and bs["p_max"] * 6 == pytest.approx(round(bs["p_max"] * 6))   # (1+#)/(B+1)
    again = Q.q3_analysis(d, r["items"]["a"], r["items"]["b"], "1PL", bootstrap=3, seed=4, points=31)["bootstrap"]
    assert again == Q.q3_analysis(d, r["items"]["a"], r["items"]["b"], "1PL", bootstrap=3, seed=4, points=31)["bootstrap"]


# ------------------------------------------------------------------------------- M2
@pytest.mark.parametrize("model,case", [("1PL", "irt1pl_ok"), ("2PL", "irt2pl_ok")])
@pytest.mark.parametrize("grid", ["gauss-hermite", "mirt"])
def test_m2_jacobian_and_df_match_finite_differences(model, case, grid):
    d = _data(case).to_numpy(float); n, k = d.shape
    r = fit_irt(pd.DataFrame(d), model=model, points=41, max_iter=500)
    a, b = r["items"]["a"].to_numpy(), r["items"]["b"].to_numpy()
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    theta, w = GF.quadrature(grid, 41)

    def margins(par):
        aa = np.ones(k) if model == "1PL" else par[k:]
        P = expit(aa * (theta[:, None] - par[:k]))
        return np.r_[w @ P, [w @ (P[:, i] * P[:, j]) for i, j in pairs]]

    par = b if model == "1PL" else np.r_[b, a]
    jac = np.column_stack([(margins(par + e) - margins(par - e)) / 2e-6 for e in np.eye(len(par)) * 1e-6])
    ours = GF.m2_statistic(d, a, b, model, grid=grid)
    np.testing.assert_allclose(ours["jacobian"], jac, atol=1e-8)
    assert ours["df"] == k + k * (k - 1) // 2 - (k if model == "1PL" else 2 * k) and ours["free_parameters"] == len(par)


def test_m2_unavailable_cases():
    d = _data("irt2pl_missing")
    assert "eksiksiz" in GF.m2_statistic(d, np.ones(10), np.zeros(10), "2PL")["reason"]
    small = _data("irt2pl_ok").iloc[:, :3]
    assert "Serbestlik" in GF.m2_statistic(small, np.ones(3), np.zeros(3), "2PL")["reason"]   # df = 6 - 6
    g = GF.global_fit(_data("irt2pl_ok"), np.ones(10), np.zeros(10), "2PL", np.zeros(1200), "EAP", converged=False)
    assert not g["m2"]["available"] and g["classification"].iloc[0]["Sınıf"] == "Hesaplanmadı"


# ------------------------------------------------------------------- R mirt references
SINGLE = {"irt1pl_ok": "1PL", "irt2pl_ok": "2PL", "irt2pl_ld": "2PL", "irt2pl_missing": "2PL"}


def _mirt_ab(case, cols, prefix="par"):
    p = _r(case, prefix)
    a = p[p.col == "a1"].set_index("row").value.loc[cols].to_numpy()
    d = p[p.col == "d"].set_index("row").value.loc[cols].to_numpy()
    return a, -d / a, d


@pytest.mark.parametrize("case", list(SINGLE))
def test_single_group_estimates_loglik_and_se_match_mirt(case):
    model = SINGLE[case]; d = _data(case); cols = list(d.columns); k = len(cols)
    a_m, b_m, d_m = _mirt_ab(case, cols)
    f = M.fit_single(d, model, quadrature="mirt")
    design = M.Design(model, ["anchor"] * k, 1); a, b = design.unpack(f["params"], 0)
    assert f["converged"] and np.abs(a - a_m).max() < 1e-5 and np.abs(b - b_m).max() < 1e-5
    x, obs = M.as_matrix(d)
    par = np.r_[[v for i in range(k) for v in ((a_m[i], b_m[i]) if model == "2PL" else (b_m[i],))]]
    assert M.loglik_at(x, obs, np.zeros(len(x)), design, par) == pytest.approx(
        float(_r(case, "loglik_fixed").value.iloc[0]), rel=1e-12)
    V = _r(case, "vcov").pivot(index="row", columns="col", values="value"); pn = _r(case, "parnum")
    num = lambda c, nm: int(pn[(pn.row == c) & (pn.col == nm)].value.iloc[0])
    for i, c in enumerate(cols):
        nd = f"d.{num(c, 'd')}"
        if model == "2PL":
            na = f"a1.{num(c, 'a1')}"; g = np.array([d_m[i] / a_m[i] ** 2, -1 / a_m[i]])
            se_b = np.sqrt(g @ V.loc[[na, nd], [na, nd]].to_numpy() @ g)
            assert np.sqrt(f["cov"][design.a[0, i], design.a[0, i]]) == pytest.approx(np.sqrt(V.loc[na, na]), rel=1e-5)
        else:
            se_b = np.sqrt(V.loc[nd, nd])
        assert np.sqrt(f["cov"][design.b[0, i], design.b[0, i]]) == pytest.approx(se_b, rel=1e-5)


@pytest.mark.parametrize("case", list(SINGLE))
def test_wle_eap_and_q3_match_mirt_with_fixed_parameters(case):
    d = _data(case); cols = list(d.columns); x, obs = M.as_matrix(d)
    it = pd.read_csv(F / f"py_items_{case}.csv"); th = pd.read_csv(F / f"py_persons_{case}.csv").theta.to_numpy()
    a, b = it.a.to_numpy(), it.b.to_numpy()
    assert np.abs(Q.wle(x, obs, a, b)[0] - _r(case, "wle").value.to_numpy()).max() < 1e-5      # mirt optimizer tol
    grid = np.linspace(-6, 6, 61); lw = -grid ** 2 / 2; lw = lw - np.log(np.exp(lw).sum())
    assert np.abs(Q.eap(x, obs, a, b, grid=(grid, lw))[0] - _r(case, "eap").value.to_numpy()).max() < 1e-6
    q3, _ = Q.q3_matrix(x, obs, th, a, b, min_pairs=1)
    for _, r in _r(case, "q3").iterrows():
        assert q3[cols.index(r.row), cols.index(r.col)] == pytest.approx(r.value, abs=1e-12)


@pytest.mark.parametrize("case", ["irt1pl_ok", "irt2pl_ok", "irt2pl_ld"])
def test_m2_matches_mirt(case):
    model = SINGLE[case]; d = _data(case); a, b, _ = _mirt_ab(case, list(d.columns))
    ours = GF.m2_statistic(d, a, b, model, grid="mirt"); ref = _r(case, "m2").set_index("col").value
    assert ours["M2"] == pytest.approx(ref["M2"], rel=1e-9) and ours["df"] == int(ref["df"])
    assert ours["p"] == pytest.approx(ref["p"], rel=1e-8) and ours["SRMSR"] == pytest.approx(ref["SRMSR"], rel=1e-9)
    n = len(d)                                                   # RMSEA2: ours N, mirt N - 1
    assert np.sqrt(max(ref["M2"] - ref["df"], 0) / ((n - 1) * ref["df"])) == pytest.approx(ref["RMSEA"], rel=1e-8, abs=1e-12)


@pytest.mark.parametrize("case,model", [("dif1pl_uniform", "1PL"), ("dif2pl_mixed", "2PL"), ("dif2pl_none", "2PL")])
def test_round0_lr_statistics_match_mirt(case, model):
    x, obs, g, cols = _xg(case); k = len(cols)
    fitter = D._Fitter(x, obs, g, model, "mirt", 41)
    base = fitter(["anchor"] * k)
    assert base["loglik"] >= float(_r(case, "lr_base_loglik").value.iloc[0]) - 1e-6
    uni = _r(case, "lr_uniform").set_index("row").value
    omni = _r(case, "lr_omnibus").set_index("row").value if model == "2PL" else uni
    via_dif = _r(case, "mirt_DIF_X2").set_index("row").value                     # mirt DIF(scheme = 'drop')
    tol = 1e-4 if model == "1PL" else 1e-5                                       # mirt EM tolerance
    for i, c in enumerate(cols):
        t = ["anchor"] * k; t[i] = "b"
        lr_u = 2 * (fitter(t)["loglik"] - base["loglik"])
        assert lr_u == pytest.approx(uni[c], abs=tol)
        if model == "2PL":
            t[i] = "ab"; lr_o = 2 * (fitter(t)["loglik"] - base["loglik"])
        else:
            lr_o = lr_u
        assert lr_o == pytest.approx(omni[c], abs=tol) and lr_o == pytest.approx(via_dif[c], abs=tol)


@pytest.mark.parametrize("case,model", [("dif1pl_uniform", "1PL"), ("dif2pl_mixed", "2PL"), ("dif2pl_none", "2PL")])
def test_final_multigroup_model_matches_mirt(case, model):
    x, obs, g, cols = _xg(case)
    anchors = pd.read_csv(F / f"py_dif_anchors_{case}.csv").anchor.tolist()
    design = M.Design(model, ["anchor" if c in anchors else ("ab" if model == "2PL" else "b") for c in cols], 2)
    f = M.fit(x, obs, g, design, "mirt")
    aR, bR, dR = _mirt_ab(case, cols, "final_par_R"); aF, bF, dF = _mirt_ab(case, cols, "final_par_F")
    pf = _r(case, "final_par_F")
    par = np.array([((aR, bR) if gg <= 0 else (aF, bF))[0 if kind == "a" else 1][i] for kind, gg, i in design.labels]
                   + [float(pf[pf.col == "MEAN_1"].value.iloc[0]), .5 * np.log(float(pf[pf.col == "COV_11"].value.iloc[0]))])
    # identical likelihood at fixed parameters; ours at least as high at its optimum
    assert M.loglik_at(x, obs, g, design, par) == pytest.approx(float(_r(case, "final_loglik_fixed_sum").value.iloc[0]), rel=1e-12)
    assert f["converged"] and f["loglik"] >= float(_r(case, "final_loglik_fixed_sum").value.iloc[0])
    assert np.abs(f["params"] - par).max() < 2e-3                                 # mirt EM stops early (sigma_F)
    V = _r(case, "final_vcov").pivot(index="row", columns="col", values="value")
    nR, nF = _r(case, "final_parnum_R"), _r(case, "final_parnum_F")
    pn = lambda n, c, nm: int(n[(n.row == c) & (n.col == nm)].value.iloc[0])
    for i, c in enumerate(cols):
        if c in anchors:
            continue
        db, se_b = D._contrast(f, i, "b")
        if model == "1PL":
            r_, f_ = f"d.{pn(nR, c, 'd')}", f"d.{pn(nF, c, 'd')}"
            se_m = np.sqrt(V.loc[r_, r_] + V.loc[f_, f_] - 2 * V.loc[r_, f_])
        else:
            names = [f"a1.{pn(nR, c, 'a1')}", f"d.{pn(nR, c, 'd')}", f"a1.{pn(nF, c, 'a1')}", f"d.{pn(nF, c, 'd')}"]
            Vs = V.loc[names, names].to_numpy(); ga = np.array([-1, 0, 1, 0.])
            gb = np.array([-dR[i] / aR[i] ** 2, 1 / aR[i], dF[i] / aF[i] ** 2, -1 / aF[i]])
            se_m = np.sqrt(gb @ Vs @ gb)
            da, se_a = D._contrast(f, i, "a")
            assert da == pytest.approx(aF[i] - aR[i], abs=1e-4) and se_a == pytest.approx(np.sqrt(ga @ Vs @ ga), rel=1e-4)
        assert db == pytest.approx(bF[i] - bR[i], abs=1e-4) and se_b == pytest.approx(se_m, rel=1e-4)
