"""Observed-score DIF: independent cross-checks (statsmodels), corrections and decision logic."""
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multitest import multipletests

from core.dif_classical import (adjust_pvalues, dif_screen, ets_mh_class, jodoin_gierl, logistic_dif,
                                mantel_haenszel, mantel_ordinal)


def binary(n=1200, k=12, dif_item=3, shift=.6, mu=0., seed=0):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 2, n); th = rng.standard_normal(n) + mu * g
    b = np.tile(np.linspace(-1.2, 1.2, k), (n, 1)); b[g == 1, dif_item] += shift
    x = (rng.random((n, k)) < expit(th[:, None] - b)).astype(int)
    return pd.DataFrame(x, columns=[f"M{j}" for j in range(k)]), g


@pytest.mark.parametrize("method,ref", [("BH", "fdr_bh"), ("BY", "fdr_by"), ("Holm", "holm"),
                                        ("Bonferroni", "bonferroni")])
def test_adjustments_match_statsmodels(method, ref):
    p = np.random.default_rng(1).random(17) ** 2; p[[2, 9]] = np.nan
    ok = np.isfinite(p)
    assert np.allclose(adjust_pvalues(p, method)[ok], multipletests(p[ok], method=ref)[1], atol=1e-14)
    assert np.isnan(adjust_pvalues(p, method)[~ok]).all()
    assert np.array_equal(adjust_pvalues(p, "none")[ok], p[ok])
    with pytest.raises(ValueError):
        adjust_pvalues(p, "xyz")


def test_mantel_haenszel_matches_statsmodels_stratified_table():
    data, g = binary(seed=2)
    x = data.to_numpy(); y = x[:, 3]; m = x.sum(1)
    r = mantel_haenszel(y, g, m)
    tabs = []
    for s in np.unique(m):
        ix = m == s
        t = np.array([[(y[ix & (g == 0)] == 1).sum(), (y[ix & (g == 0)] == 0).sum()],
                      [(y[ix & (g == 1)] == 1).sum(), (y[ix & (g == 1)] == 0).sum()]])
        if t[0].sum() and t[1].sum() and t[:, 0].sum() and t[:, 1].sum():
            tabs.append(t)
    st = StratifiedTable(tabs)
    assert np.isclose(r["chi2"], st.test_null_odds(correction=True).statistic, rtol=1e-12)
    assert np.isclose(r["alpha_mh"], st.oddsratio_pooled, rtol=1e-12)
    assert np.isclose(r["se_log_or"], st.logodds_pooled_se, rtol=1e-12)
    assert r["delta"] < -1  # item harder for focal: negative ETS delta


def test_mantel_ordinal_equals_generalized_cmh_correlation_statistic():
    rng = np.random.default_rng(3); n = 600
    g = rng.integers(0, 2, n); y = rng.integers(0, 4, n); m = rng.integers(0, 5, n)
    r = mantel_ordinal(y, g, m)
    # independent: CMH "nonzero correlation" M^2 with row scores g and column scores y (Agresti 2002, 7.5)
    num = var = 0.
    for s in np.unique(m):
        ix = m == s; T = ix.sum(); gs, ys = g[ix], y[ix]
        num += (gs * ys).sum() - gs.sum() * ys.sum() / T
        var += (T * (gs ** 2).sum() - gs.sum() ** 2) * (T * (ys ** 2).sum() - ys.sum() ** 2) / (T * T * (T - 1))
    assert np.isclose(r["chi2"], num ** 2 / var, rtol=1e-10)


def test_logistic_dif_detects_planted_uniform_dif_and_lr_identities():
    data, g = binary(n=2000, seed=4, shift=.8)
    x = data.to_numpy(); m = x.sum(1)
    r = logistic_dif(x[:, 3], g, m)
    assert r["p_uniform"] < 1e-4 and r["beta_group"] < 0
    assert np.isclose(r["lr_omnibus"], r["lr_uniform"] + r["lr_nonuniform"], atol=1e-8)
    null = logistic_dif(x[:, 6], g, m)
    assert null["p_omnibus"] > .001 and null["delta_r2"] < .035


def test_dif_screen_purification_and_classes():
    data, g = binary(n=3000, seed=5, shift=1.0)
    frame, meta = dif_screen(data, g, 0, 1, method="MH", adjust="Holm")
    assert frame.loc[3, "DIF (düzeltilmiş p < alpha)"] and not frame.loc[3, "Eşleştirmede"]
    assert frame.drop(index=3)["Eşleştirmede"].all()
    assert frame.loc[3, "ETS sınıfı"] in ("B", "C") and meta["rounds"] >= 1
    lr, _ = dif_screen(data, g, 0, 1, method="LR")
    assert lr.loc[3, "DIF (düzeltilmiş p < alpha)"]
    assert set(lr["Jodoin–Gierl sınıfı"]) <= {"A", "B", "C"}


def test_ordinal_methods_run_and_flag_shift():
    rng = np.random.default_rng(6); n, k = 1500, 8
    g = rng.integers(0, 2, n); th = rng.standard_normal(n)
    x = np.zeros((n, k), int)
    for j in range(k):
        thr = np.linspace(-1, 1, 3) + (0.7 * g[:, None] if j == 2 else 0)
        x[:, j] = (rng.random(n)[:, None] < expit(1.3 * (th[:, None] - thr))).sum(1)
    data = pd.DataFrame(x, columns=[f"P{j}" for j in range(k)])
    mh, _ = dif_screen(data, g, 0, 1, method="MH", ordinal=True)
    assert mh.loc[2, "DIF (düzeltilmiş p < alpha)"] and mh.loc[2, "smd"] < 0
    olr, _ = dif_screen(data, g, 0, 1, method="LR", ordinal=True)
    assert olr.loc[2, "DIF (düzeltilmiş p < alpha)"]


def test_decision_rules_and_input_validation():
    assert ets_mh_class(-2.0, .1, .001) == "C" and ets_mh_class(-1.2, .1, .001) == "B"
    assert ets_mh_class(-2.0, .1, .2) == "A" and ets_mh_class(-.5, .1, .001) == "A"
    assert jodoin_gierl(.08, .01) == "C" and jodoin_gierl(.05, .01) == "B" and jodoin_gierl(.08, .2) == "A"
    data, g = binary(seed=7)
    bad = data.astype(float); bad.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="eksik"):
        dif_screen(bad, g, 0, 1)
    with pytest.raises(ValueError):
        dif_screen(data.replace(1, 2), g, 0, 1)
    with pytest.raises(ValueError):
        dif_screen(data, g, 0, 0)
