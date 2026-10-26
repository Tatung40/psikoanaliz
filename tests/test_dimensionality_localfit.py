import numpy as np
import pandas as pd
import pytest
from scipy import stats, optimize
from scipy.stats import multivariate_normal as MVN
from statsmodels.stats.multitest import multipletests

from core import dimensionality as dm
from core import local_fit as lf


def _sim2pl(n, a, b, seed, theta=None):
    rng = np.random.default_rng(seed)
    th = rng.normal(size=n) if theta is None else theta
    p = 1 / (1 + np.exp(-a * (th[:, None] - b)))
    return pd.DataFrame((rng.random(p.shape) < p).astype(float), columns=[f"M{i+1}" for i in range(len(a))])


@pytest.mark.parametrize("h,k,r", [(0.3, -0.5, 0.6), (1.2, 0.4, -0.8), (-2, 1, 0.995), (0.5, 0.5, 0.9999), (2, -1, -0.999), (0, 0, 0.0)])
def test_bvn_matches_scipy(h, k, r):
    assert dm.bvn_cdf(h, k, r) == pytest.approx(MVN.cdf([h, k], cov=[[1, r], [r, 1]]), abs=1e-10)


def test_tetrachoric_matches_independent_scipy_ml():
    rng = np.random.default_rng(3)
    z = rng.multivariate_normal([0, 0], [[1, .45], [.45, 1]], 3000)
    x, y = (z[:, 0] > .4) * 1., (z[:, 1] > -.2) * 1.
    t = np.array([[((x == i) & (y == j)).sum() for j in (0, 1)] for i in (0, 1)])
    h, kk = stats.norm.ppf((x == 0).mean()), stats.norm.ppf((y == 0).mean())

    def nll(r):
        p00 = MVN.cdf([h, kk], cov=[[1, r], [r, 1]])
        p = np.array([[p00, stats.norm.cdf(h) - p00], [stats.norm.cdf(kk) - p00, 1 - stats.norm.cdf(h) - stats.norm.cdf(kk) + p00]])
        return -(t * np.log(p)).sum()
    ref = optimize.minimize_scalar(nll, bounds=(-.99, .99), method="bounded", options={"xatol": 1e-8}).x
    assert dm.polychoric_pair(x, y)[0] == pytest.approx(ref, abs=1e-4)


def test_polychoric_recovers_population_rho():
    rng = np.random.default_rng(5)
    z = rng.multivariate_normal([0, 0], [[1, .6], [.6, 1]], 20000)
    cut = [-1, -.2, .5, 1.3]
    rho, n, flag = dm.polychoric_pair(np.digitize(z[:, 0], cut) * 1., np.digitize(z[:, 1], cut) * 1.)
    assert rho == pytest.approx(.6, abs=.02) and flag == ""


def test_type_detection_and_invalid_methods():
    assert dm.data_type(pd.DataFrame({"a": [0, 1, 1]})) == "binary"
    assert dm.data_type(pd.DataFrame({"a": [1, 2, 3, 4]})) == "ordinal"
    cont = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 3)))
    assert dm.data_type(cont) == "continuous"
    with pytest.raises(ValueError):
        dm.correlation_matrix(cont, "polychoric")


def test_paf_recovers_population_loadings_exactly():
    lam = np.array([.8, .7, .6, .5, .4, .7])
    R = np.outer(lam, lam); np.fill_diagonal(R, 1)
    L, h2, conv, hey = dm.paf(R, 1, tol=1e-12, max_iter=5000)
    assert conv and not hey
    np.testing.assert_allclose(np.abs(L[:, 0]), lam, atol=1e-5)


def test_kmo_matches_regression_partial_correlations():
    rng = np.random.default_rng(1)
    X = rng.multivariate_normal(np.zeros(4), [[1, .5, .4, .3], [.5, 1, .45, .2], [.4, .45, 1, .35], [.3, .2, .35, 1]], 500)
    R = np.corrcoef(X.T)
    k = 4; P = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i == j: continue
            o = [m for m in range(k) if m not in (i, j)]
            Z = np.c_[np.ones(len(X)), X[:, o]]
            ri = X[:, i] - Z @ np.linalg.lstsq(Z, X[:, i], rcond=None)[0]
            rj = X[:, j] - Z @ np.linalg.lstsq(Z, X[:, j], rcond=None)[0]
            P[i, j] = np.corrcoef(ri, rj)[0, 1]
    R0 = R - np.eye(k)
    ref = (R0 ** 2).sum() / ((R0 ** 2).sum() + (P ** 2).sum())
    assert dm.kmo(R)[0] == pytest.approx(ref, abs=1e-10)


def test_bartlett_identity_and_formula():
    stat, df, p = dm.bartlett(np.eye(5), 200)
    assert stat == pytest.approx(0, abs=1e-12) and df == 10 and p == pytest.approx(1)
    R = np.full((3, 3), .5); np.fill_diagonal(R, 1)
    stat, df, p = dm.bartlett(R, 101)
    assert stat == pytest.approx(-(100 - 11 / 6) * np.log(np.linalg.det(R)))


def test_parallel_analysis_one_vs_two_factors():
    rng = np.random.default_rng(11)
    one = _sim2pl(800, np.full(10, 1.6), np.linspace(-1.5, 1.5, 10), 2)
    th1, th2 = rng.normal(size=(2, 800))
    a = np.full(6, 2.0); b = np.linspace(-1, 1, 6)
    two = pd.concat([_sim2pl(800, a, b, 3, th1), _sim2pl(800, a, b, 4, th2).add_prefix("B")], axis=1)
    assert dm.parallel_analysis(one, "tetrachoric", reps=20)[1] == 1
    res = dm.dimensionality_analysis(two, pa_reps=20, rotation="promax")
    assert res["n_factors"] == 2
    L = res["loadings"][["F1", "F2"]].abs().to_numpy()
    first = L[:6].argmax(1); second = L[6:].argmax(1)
    assert len(set(first)) == 1 and len(set(second)) == 1 and first[0] != second[0]
    assert abs(res["factor_correlations"].iloc[0, 1]) < .2


def test_bh_matches_statsmodels():
    p = np.array([.001, .2, np.nan, .03, .04, .5])
    m = np.isfinite(p)
    np.testing.assert_allclose(lf.bh_fdr(p)[m], multipletests(p[m], method="fdr_bh")[1])
    assert np.isnan(lf.bh_fdr(p)[2])


def test_local_dependence_pair_detected():
    rng = np.random.default_rng(21)
    df = _sim2pl(1000, np.full(10, 1.4), np.linspace(-1.5, 1.5, 10), 7)
    flip = rng.random(1000) < .35
    df.loc[flip, "M10"] = df.loc[flip, "M9"]
    res = lf.run(df, "2PL")
    top = res["local"]["pairs"].iloc[0]
    assert {top["Madde 1"], top["Madde 2"]} == {"M9", "M10"}
    assert res["local"]["pairs"]["Artık korelasyon (1F)"].abs().idxmax() == 0


def test_misfitting_item_flagged_with_fdr():
    rng = np.random.default_rng(31)
    df = _sim2pl(1500, np.full(12, 1.3), np.linspace(-1.5, 1.5, 12), 9)
    th = df.sum(axis=1).to_numpy()
    df["M12"] = (rng.random(1500) < np.where(np.abs(th - 6) < 2, .85, .2)).astype(float)
    res = lf.run(df, "2PL")
    t = res["itemfit"].set_index("Madde")
    assert t.loc["M12", "Karar"] == "Uyumsuz (FDR)"
    assert t["p"].idxmin() == "M12"
    assert res["certified"] and len(res["obs_exp"]["M12"]) >= 2


def test_3pl_inference_not_certified_and_polytomous_disabled():
    df = _sim2pl(800, np.full(8, 1.4), np.linspace(-1, 1, 8), 13)
    tab, groups, cert = lf.item_fit(df, lf.fit_irt(df, "3PL")["items"], "3PL")
    assert not cert and (tab["Sertifika"] == lf.EXPERIMENTAL).all() and (tab["Karar"] == "Karar üretilmedi").all()
    with pytest.raises(ValueError):
        lf.run(pd.DataFrame(np.random.default_rng(0).integers(0, 4, (100, 5)).astype(float)), "2PL")


def test_no_fdr_flags_under_true_2pl():
    df = _sim2pl(1500, np.linspace(.8, 2, 12), np.linspace(-1.5, 1.5, 12), 41)
    res = lf.run(df, "2PL")
    assert (res["itemfit"]["Karar"] == "Uyumsuz (FDR)").sum() == 0
