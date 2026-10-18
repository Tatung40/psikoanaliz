"""B9: DIF, Q3 and global fit for the dichotomous Rasch model.

Reference fixtures: tests/fixtures/rasch_b9/r_b9_results.csv (scripts/mirt_rasch_b9_reference.R;
R 4.3.3, mirt 1.48, eRm 1.0-4) built from Python inputs (scripts/rasch_b9_inputs_for_r.py).
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
import core.rasch as R
import core.rasch_dif as D
import core.rasch_q3 as Q
import core.rasch_globalfit as GF

F8 = Path(__file__).parent / "fixtures" / "rasch"
F9 = Path(__file__).parent / "fixtures" / "rasch_b9"


@lru_cache(maxsize=None)
def _data(case):
    folder = F9 if (F9 / f"data_{case}.csv").exists() else F8
    return pd.read_csv(folder / f"data_{case}.csv", na_values="NA")


@lru_cache(maxsize=None)
def _ref():
    return pd.read_csv(F9 / "r_b9_results.csv", keep_default_na=False, na_values=["NA"])


def _r(case, quantity):
    r = _ref()
    return r[(r.case == case) & (r.quantity == quantity)]


@lru_cache(maxsize=None)
def _dif(case):
    d = _data(case)
    return D.dif_analysis(d.drop(columns="Grup"), d["Grup"], "R", "F")


def _xg(case):
    d = _data(case)
    x, obs = R._as_matrix(d.drop(columns="Grup"))
    return x, obs, (d["Grup"] == "F").to_numpy().astype(int), list(d.columns[1:])


# ---------------------------------------------------------------- DIF: mathematics
@pytest.mark.parametrize("quadrature", ["gauss-hermite", "mirt"])
def test_multigroup_score_matches_finite_differences(quadrature):
    x, obs, g, _ = _xg("dif_known")
    idx_r, idx_f, labels, latent = D._design(x.shape[1], [2, 7])
    G = D._Groups(x, obs, g, quadrature, 41)
    p = np.r_[np.linspace(-1.5, 1.5, latent), -.3, .05, .1]
    val, grad = D._negloglik(p, G, idx_r, idx_f, latent, len(x))
    fd = np.array([(D._negloglik(p + e, G, idx_r, idx_f, latent, len(x))[0]
                    - D._negloglik(p - e, G, idx_r, idx_f, latent, len(x))[0]) / 2e-6 for e in np.eye(len(p)) * 1e-6])
    assert np.abs(grad - fd).max() <= 1e-6 * np.abs(grad).max()


def _mirt_vector(case, studied, labels, latent):
    cols = _xg(case)[3]
    pr, pf = _r(case, "par_R"), _r(case, "par_F")
    b_r = -pr[pr.col == "d"].set_index("row").value.loc[cols].to_numpy()
    b_f = -pf[pf.col == "d"].set_index("row").value.loc[cols].to_numpy()
    p = np.zeros(latent + 3)
    for j, (grp, i) in enumerate(labels):
        p[j] = b_f[i] if grp == "F" else b_r[i]
    p[latent] = float(pf[pf.col == "MEAN_1"].value.iloc[0])
    p[latent + 1] = .5 * np.log(float(pr[pr.col == "COV_11"].value.iloc[0]))
    p[latent + 2] = .5 * np.log(float(pf[pf.col == "COV_11"].value.iloc[0]))
    return p, b_r, b_f


@pytest.mark.parametrize("case", ["dif_known", "dif_none"])
def test_multigroup_loglik_equals_mirt_at_fixed_parameters(case):
    x, obs, g, cols = _xg(case)
    anchors = pd.read_csv(F9 / f"py_dif_anchors_{case}.csv").anchor.tolist()
    studied = [i for i, c in enumerate(cols) if c not in anchors]
    idx_r, idx_f, labels, latent = D._design(len(cols), studied)
    p, _, _ = _mirt_vector(case, studied, labels, latent)
    ours = -D._negloglik(p, D._Groups(x, obs, g, "mirt", 41), idx_r, idx_f, latent, len(x))[0] * len(x)
    assert ours == pytest.approx(float(_r(case, "loglik_fixed_sum").value.iloc[0]), rel=1e-12)
    # mirt's reported EM logLik is not evaluated exactly at its final parameters
    assert abs(ours - float(_r(case, "loglik").value.iloc[0])) > 1e-5


def test_identification_matches_mirt_multigroup():
    for case in ("dif_known", "dif_none"):
        assert _r(case, "est_R").set_index("col").value.to_dict() == {"MEAN_1": 0., "COV_11": 1.}
        assert _r(case, "est_F").set_index("col").value.to_dict() == {"MEAN_1": 1., "COV_11": 1.}


def test_wald_contrast_and_se_match_mirt_multipleGroup():
    case = "dif_known"
    x, obs, g, cols = _xg(case)
    anchors = pd.read_csv(F9 / f"py_dif_anchors_{case}.csv").anchor.tolist()
    studied = [i for i, c in enumerate(cols) if c not in anchors]
    fit = D.fit_multigroup(x, obs, g, studied, quadrature="mirt")
    _, b_r, b_f = _mirt_vector(case, studied, *D._design(len(cols), studied)[2:4])
    assert fit["converged"]
    assert fit["loglik"] >= float(_r(case, "loglik_fixed_sum").value.iloc[0])     # ours at least as high
    assert np.abs(fit["b_R"] - b_r).max() <= 1e-4 and np.abs(fit["b_F"] - b_f).max() <= 1e-4
    vc = _r(case, "vcov").pivot(index="row", columns="col", values="value")
    n_r = _r(case, "parnum_R").set_index("row").value; n_f = _r(case, "parnum_F").set_index("row").value
    for i in studied:
        a, b = f"d.{int(n_r[cols[i]])}", f"d.{int(n_f[cols[i]])}"
        se_m = np.sqrt(vc.loc[a, a] + vc.loc[b, b] - 2 * vc.loc[a, b])
        delta, se = D.item_contrast(fit, i)
        assert delta == pytest.approx(b_f[i] - b_r[i], abs=1e-4)
        assert se == pytest.approx(se_m, rel=1e-4)


# --------------------------------------------------------------- DIF: procedure
def test_known_dif_detected_and_impact_not_mistaken_for_dif():
    known, none = _dif("dif_known"), _dif("dif_none")
    flagged = set(known["table"].loc[known["table"]["DIF işareti"], "Madde"])
    assert flagged == {"M3", "M8"} and known["stable"]
    t = known["table"].set_index("Madde")
    assert t.loc["M3", "Δ (b_odak − b_ref)"] > 0 > t.loc["M8", "Δ (b_odak − b_ref)"]          # sign convention
    assert not none["table"]["DIF işareti"].any()
    assert none["joint"]["mu_F"] < -.3                                                     # impact absorbed
    assert known["anchors"] == pd.read_csv(F9 / "py_dif_anchors_dif_known.csv").anchor.tolist()


def test_naive_unaligned_difference_is_not_the_reported_contrast():
    """Separate calibrations (each with latent mean 0) confound impact with DIF."""
    d = _data("dif_none")
    sep = {g: R.fit_rasch(d.loc[d.Grup == g].drop(columns="Grup"))["b"] for g in ("R", "F")}
    naive = sep["F"] - sep["R"]
    aligned = _dif("dif_none")["table"]["Δ (b_odak − b_ref)"].to_numpy()
    assert naive.mean() > .35                     # shifted by the latent impact (focal mean -0.5)
    assert abs(aligned.mean()) < .1 and np.abs(aligned).max() < .35


def test_purification_removes_dif_items_from_anchor_set():
    k = _dif("dif_known")
    assert "M3" not in k["anchors"] and "M8" not in k["anchors"]
    assert list(k["history"]["Tur"]) == [0, 1] and k["history"]["İşaretlenen"].iloc[0] == ["M3", "M8"]
    no_pur = D.dif_analysis(_data("dif_known").drop(columns="Grup"), _data("dif_known")["Grup"], "R", "F", purify=False)
    assert len(no_pur["history"]) == 1 and no_pur["purify"] is False


def test_bh_and_ets_helpers():
    q = D.benjamini_hochberg([.01, .04, .03, np.nan, .5])
    # m=4 finite p: sorted .01,.03,.04,.5 -> .04,.0533,.0533,.5 (step-up)
    assert np.allclose(q[[0, 1, 2, 4]], [.04, .04 * 4 / 3, .04 * 4 / 3, .5], atol=1e-9)
    assert np.isnan(q[3])
    assert D.ets_class(.7, .05, True) == "C (büyük)" and D.ets_class(.5, .05, True) == "B (orta)"
    assert D.ets_class(.7, .05, False).startswith("A") and D.ets_class(.3, .05, True).startswith("A")


def test_small_unequal_groups_and_invalid_settings():
    d = _data("dif_known")
    small = pd.concat([d[d.Grup == "R"].iloc[:300], d[d.Grup == "F"].iloc[:60]])
    out = D.dif_analysis(small.drop(columns="Grup"), small["Grup"], "R", "F")
    assert any("küçük" in w for w in out["warnings"]) and out["sizes"] == {"R": 300, "F": 60}
    with pytest.raises(ValueError, match="en az 30"):
        tiny = pd.concat([d[d.Grup == "R"].iloc[:100], d[d.Grup == "F"].iloc[:20]])
        D.dif_analysis(tiny.drop(columns="Grup"), tiny["Grup"], "R", "F")
    with pytest.raises(ValueError, match="farklı"):
        D.dif_analysis(d.drop(columns="Grup"), d["Grup"], "R", "R")
    with pytest.raises(ValueError, match="Anlamlılık"):
        D.dif_analysis(d.drop(columns="Grup"), d["Grup"], "R", "F", alpha=.7)


def test_untestable_item_and_disconnected_design():
    d = _data("dif_known").copy()
    d.loc[d.Grup == "F", "M5"] = 1.                                                         # constant in focal
    out = D.dif_analysis(d.drop(columns="Grup"), d["Grup"], "R", "F")
    row = out["table"].set_index("Madde").loc["M5"]
    assert row["Etki sınıfı (ETS benzeri)"] == "Test edilemedi" and "aynı" in row["Not"]
    dis = _data("dif_known").copy()
    items = list(dis.columns[1:])
    dis.loc[dis.Grup == "R", items[:6]] = np.nan; dis.loc[dis.Grup == "F", items[6:]] = np.nan
    with pytest.raises(ValueError, match="Bağlantısız"):
        D.dif_analysis(dis.drop(columns="Grup"), dis["Grup"], "R", "F")


def test_erm_cml_wald_agreement_indicator_only():
    for case in ("dif_known", "dif_none"):
        z_erm = _r(case, "erm_wald"); z_erm = z_erm[z_erm.col == "z"].set_index("row").value.loc[_xg(case)[3]]
        assert np.corrcoef(z_erm, _dif(case)["table"]["z"])[0, 1] > .99
    assert float(_r("dif_known", "erm_andersen_lr").set_index("col").value["p"]) < .01
    assert float(_r("dif_none", "erm_andersen_lr").set_index("col").value["p"]) > .05


# ----------------------------------------------------------------------- Q3
@pytest.mark.parametrize("case", ["ld_pair", "rasch_missing"])
def test_q3_matches_mirt_given_same_residuals(case):
    d = _data(case); x, obs = R._as_matrix(d); cols = list(d.columns)
    b = pd.read_csv(F9 / f"py_q3_items_{case}.csv").b.to_numpy()
    th = pd.read_csv(F9 / f"py_q3_persons_{case}.csv").theta.to_numpy()
    q3, counts, _ = Q.q3_matrix(x, obs, th, b, min_pairs=1)
    for _, r in _r(case, "q3").iterrows():
        assert q3[cols.index(r.row), cols.index(r.col)] == pytest.approx(r.value, abs=1e-12)


def test_q3_adjusted_ranking_detects_planted_pair():
    out = Q.q3_analysis(_data("ld_pair"))
    top = out["pairs"].iloc[0]
    assert {top["Madde 1"], top["Madde 2"]} == {"M2", "M3"}
    off = out["q3"].to_numpy()[np.triu_indices(10, 1)]
    assert out["mean_q3"] == pytest.approx(off.mean())
    np.testing.assert_allclose(out["adjusted"].to_numpy()[np.triu_indices(10, 1)], off - off.mean())
    assert out["expected_under_li"] == pytest.approx(-1 / 9)


def test_q3_missing_pairs_and_zero_variance_are_undefined():
    x = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 0, 1]], float); obs = np.ones_like(x, bool)
    obs[0, 1] = False
    theta = np.zeros(4); b = np.array([0., 0., -50.])                                     # item 3: P≈1 -> zero variance
    q3, counts, _ = Q.q3_matrix(x, obs, theta, b, min_pairs=1)
    assert counts[0, 1] == 3 and np.isfinite(q3[0, 1]) and np.isnan(q3[0, 2])
    q3b, _, _ = Q.q3_matrix(x, obs, theta, b, min_pairs=4)
    assert np.isnan(q3b[0, 1])                                                             # too few pairs


def test_q3_bootstrap_reference_and_settings():
    out = Q.q3_analysis(_data("ld_pair"), bootstrap=20, seed=11)
    bs = out["bootstrap"]
    assert bs["used"] == 20 and 0 < bs["p_max"] <= 1 and "Bootstrap %95 (çift)" in out["pairs"]
    again = Q.q3_analysis(_data("ld_pair"), bootstrap=20, seed=11)["bootstrap"]
    assert again["p_max"] == bs["p_max"]                                                    # reproducible
    with pytest.raises(ValueError, match="Bootstrap"):
        Q.q3_analysis(_data("ld_pair"), bootstrap=10_000)
    with pytest.raises(ValueError, match="WLE veya EAP"):
        Q.q3_analysis(_data("ld_pair"), person_method="ML")


# ------------------------------------------------------------------ global fit
@pytest.mark.parametrize("case", ["ld_pair", "rasch_main"])
def test_m2_matches_mirt_with_fixed_sd(case):
    d = _data(case); cols = list(d.columns); p = _r(case, "m2_par")
    b = -p[p.col == "d"].set_index("row").value.loc[cols].to_numpy()
    sd = float(np.sqrt(p[p.col == "COV_11"].value.iloc[0]))
    ours = GF.m2_statistic(d, b, sd, grid="mirt", free_sd=False)
    ref = _r(case, "m2_fixed_sd").set_index("col").value
    assert ours["M2"] == pytest.approx(ref["M2"], rel=1e-9) and ours["df"] == int(ref["df"])
    assert ours["SRMSR"] == pytest.approx(ref["SRMSR"], rel=1e-9)
    assert ours["p"] == pytest.approx(ref["p"], rel=1e-6, abs=1e-12)
    n = len(d)                                                    # RMSEA2: ours uses N, mirt N - 1
    assert ours["RMSEA2"] == pytest.approx(np.sqrt(max(ours["M2"] - ours["df"], 0) / (n * ours["df"])))
    assert np.sqrt(max(ref["M2"] - ref["df"], 0) / ((n - 1) * ref["df"])) == pytest.approx(ref["RMSEA"], rel=1e-6)


def test_m2_free_sd_projects_latent_sd_unlike_mirt():
    case = "ld_pair"; d = _data(case); cols = list(d.columns); p = _r(case, "m2_par")
    b = -p[p.col == "d"].set_index("row").value.loc[cols].to_numpy(); sd = float(np.sqrt(p[p.col == "COV_11"].value.iloc[0]))
    ours = GF.m2_statistic(d, b, sd, grid="mirt", free_sd=True)
    free, fixed = _r(case, "m2_free").set_index("col").value, _r(case, "m2_fixed_sd").set_index("col").value
    assert free["M2"] == pytest.approx(fixed["M2"], rel=1e-12)          # mirt: sigma not in its Jacobian ...
    assert int(free["df"]) == int(fixed["df"]) - 1 == ours["df"]          # ... but subtracted from df
    assert ours["M2"] < fixed["M2"]                                        # ours projects sigma out


def test_m2_unavailable_with_missing_data_and_global_fit_classes():
    a = R.rasch_analysis(_data("rasch_missing"))
    g = GF.global_fit(a)
    assert not g["m2"]["available"] and "eksiksiz" in g["m2"]["reason"]
    assert g["raw_incomplete_n"] > 0 and g["raw_complete_n"] + g["raw_incomplete_n"] == a["fit"]["n"]
    cls = g["classification"].set_index("İstatistik")["Sınıf"]
    assert cls["M2, RMSEA2"] == "Hesaplanmadı" and cls["Infit/Outfit (B8)"] == GF.CLASS_DESCRIPTIVE
    full = GF.global_fit(R.rasch_analysis(_data("rasch_main")))
    assert full["m2"]["available"] and full["m2"]["p"] < .001                 # misfitting items detected
    assert full["classification"].set_index("İstatistik").loc["M2, RMSEA2", "Sınıf"] == GF.CLASS_INFERENTIAL


def test_raw_score_distribution_expected_sums_to_n():
    a = R.rasch_analysis(_data("ld_pair"))
    raw, n, _ = GF.raw_score_distribution(a["fit"]["data"], a["fit"]["b"], a["fit"]["sd"])
    assert raw["Beklenen"].sum() == pytest.approx(n) and raw["Gözlenen"].sum() == n
    strata = GF.item_strata(a["fit"]["data"], a["fit"]["observed"], a["persons"]["WLE θ"].to_numpy(),
                            a["fit"]["b"], list(a["items"]["Madde"]))
    assert set(strata["Tabaka"]) == {1, 2, 3, 4, 5}


@pytest.mark.parametrize("grid", ["gauss-hermite", "mirt"])
def test_m2_free_sd_jacobian_matches_finite_differences(grid):
    """Independent M2: margins differentiated numerically in (b, sigma)."""
    d = _data("ld_pair").to_numpy(float); n, k = d.shape
    f = R.fit_rasch(d); b, sd = f["b"], f["sd"]
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]

    def margins(par):
        theta, w, _ = GF._quadrature(par[-1], grid, 41)
        P = expit(theta[:, None] - par[None, :k])
        return np.r_[w @ P, [w @ (P[:, i] * P[:, j]) for i, j in pairs]]

    par = np.r_[b, sd]; pi = margins(par)
    jac = np.column_stack([(margins(par + e) - margins(par - e)) / 2e-6 for e in np.eye(k + 1) * 1e-6])
    theta, w, _ = GF._quadrature(sd, grid, 41); P = expit(theta[:, None] - b[None, :])
    sets = [(i,) for i in range(k)] + pairs
    xi = np.array([[w @ np.prod(P[:, sorted(set(a) | set(c))], axis=1) for c in sets] for a in sets]) - np.outer(pi, pi)
    e = np.r_[d.mean(0), [np.mean(d[:, i] * d[:, j]) for i, j in pairs]] - pi
    W = np.linalg.inv(xi); C = W - W @ jac @ np.linalg.solve(jac.T @ W @ jac, jac.T @ W)
    ours = GF.m2_statistic(d, b, sd, grid=grid, free_sd=True)
    assert ours["M2"] == pytest.approx(n * e @ C @ e, rel=1e-6) and ours["df"] == len(sets) - k - 1
