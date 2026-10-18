"""B8: core.rasch unit tests (analytic score, WLE, fit formulas, validation, reliability)."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq
from scipy.special import expit
import core.rasch as R

FIX = Path(__file__).parent / "fixtures" / "rasch"


@lru_cache(maxsize=None)
def _data(case):
    return pd.read_csv(FIX / f"data_{case}.csv", na_values="NA")


@lru_cache(maxsize=None)
def _analysis(case):
    return R.rasch_analysis(_data(case))


@pytest.mark.parametrize("case", ["rasch_main", "rasch_missing"])
def test_analytic_score_matches_finite_differences(case):
    x, obs, _, _ = R.validate_rasch_data(_data(case))
    z, lw = R._grid(41)
    x1, x0 = np.where(obs, x, 0.), np.where(obs, 1 - x, 0.)
    k = x.shape[1]
    p = np.r_[np.linspace(-1.5, 1.5, k), np.log(1.1)]
    score = R._score(p, x1, x0, z, lw)
    fd = np.array([(R._objective(p + e, x1, x0, z, lw, 1)[0] - R._objective(p - e, x1, x0, z, lw, 1)[0]) / -2e-6
                   for e in np.eye(len(p)) * 1e-6])
    assert np.abs(score - fd).max() <= 1e-6 * np.abs(score).max()


@pytest.mark.parametrize("case", ["rasch_main", "rasch_missing"])
def test_calibration_converges_and_recovers_generating_values(case):
    out = _analysis(case)
    f = out["fit"]
    truth = pd.read_csv(FIX / f"truth_{case}.csv")
    sd_true = float(pd.read_csv(FIX / "cases.csv").set_index("case").loc[case, "sd"])
    assert f["converged"] and f["max_score_per_person"] <= 1e-6 and not f["at_bound"]
    rasch_items = truth.egim.to_numpy() == 1            # misfitting generators excluded from recovery
    err = np.abs(f["b"] - truth.b.to_numpy())[rasch_items]
    assert err.max() <= 4 * f["items"]["SH(b)"].to_numpy()[rasch_items].max()
    assert abs(f["sd"] - sd_true) <= 4 * f["se_sd"] + .05
    assert f["param_count"] == f["k"] + 1


def test_wle_solves_weighted_likelihood_equation_exactly():
    x, obs, _, _ = R.validate_rasch_data(_data("rasch_missing"))
    b = _analysis("rasch_missing")["fit"]["b"]
    theta, se = R.wle(x, obs, b)
    for i in (0, 5, 17, 400):
        m = obs[i]; bb = b[m]; r = x[i, m].sum()
        def f(t):
            p = expit(t - bb); w = p * (1 - p)
            return r - p.sum() + (w * (1 - 2 * p)).sum() / (2 * w.sum())
        assert theta[i] == pytest.approx(brentq(f, -12, 12, xtol=1e-13), abs=1e-10)
        p = expit(theta[i] - bb)
        assert se[i] == pytest.approx(1 / np.sqrt((p * (1 - p)).sum()), rel=1e-12)


def test_fit_statistics_hand_computation():
    x = np.array([[1, 0, 1], [0, 0, 1], [1, 1, 0], [1, 1, 1]], float); obs = np.ones_like(x, bool)
    obs[1, 2] = False
    theta = np.array([.3, -1., .8, 2.]); b = np.array([-.5, .4, 1.1])
    item, person = R.fit_statistics(x, obs, theta, b)
    p = expit(theta[:, None] - b); w = p * (1 - p); c = p * (1 - p) ** 4 + (1 - p) * p ** 4
    j = 0; m = obs[:, j]
    y2 = (x[m, j] - p[m, j]) ** 2
    assert item["Outfit MNSQ"][j] == pytest.approx(np.mean(y2 / w[m, j]))
    assert item["Infit MNSQ"][j] == pytest.approx(y2.sum() / w[m, j].sum())
    q = np.sqrt((c[m, j] / w[m, j] ** 2).sum() / m.sum() ** 2 - 1 / m.sum())
    assert item["Outfit ZSTD"][j] == pytest.approx((np.cbrt(item["Outfit MNSQ"][j]) - 1) * 3 / q + q / 3)
    n = 1; m = obs[n]
    assert person["Gözlem"][n] == 2
    assert person["Infit MNSQ"][n] == pytest.approx(((x[n, m] - p[n, m]) ** 2).sum() / w[n, m].sum())


def test_misfitting_generators_are_flagged():
    items = _analysis("rasch_main")["items"].set_index("Madde")
    assert items.loc["M8", "Uyum"] == "İnceleme önerilir"                       # slope 2.5: overfit
    assert items.loc["M8", "Outfit MNSQ"] < .7 and items.loc["M8", "Infit ZSTD"] < -2
    assert items.loc["M3", "Uyum"] == "İnceleme önerilir"                       # slope 0.4: underfit
    assert items.loc["M3", "Outfit MNSQ"] > 1.3 and items.loc["M3", "Outfit ZSTD"] > 2
    wide = R.rasch_analysis(_data("rasch_main"), mnsq_range=(.5, 1.5)).get("items").set_index("Madde")
    assert wide.loc["M3", "Uyum"] == "Kabul aralığında"                          # range is user-selectable


def test_extreme_persons_and_missing_reported():
    out = _analysis("rasch_missing")
    persons = out["persons"]
    ext = persons["Uç puan"].to_numpy()
    assert ext.any() and persons.loc[ext, "Infit MNSQ"].isna().all()
    assert np.isfinite(persons["WLE θ"]).all()
    assert any("MAR" in w for w in out["warnings"]) and any("uç değerde" in w for w in out["warnings"])
    assert out["fit"]["missing_cells"] == int(_data("rasch_missing").isna().sum().sum())


def test_separation_formula_and_reliability_table():
    s = R.separation(np.array([-1., 0., 1., 2.]), np.array([.5, .5, .5, .5]))
    var = np.var([-1., 0., 1., 2.])
    assert s["Güvenirlik"] == pytest.approx((var - .25) / var)
    assert s["Ayırma indeksi (G)"] == pytest.approx(np.sqrt((var - .25) / .25))
    rel = _analysis("rasch_main")["reliability"]
    assert list(rel["Gösterge"]) == ["Kişi (WLE, model SH, uç puanlar hariç)", "Madde (MML SH)"]
    assert 0 < rel["Güvenirlik"].iloc[0] < 1 and 0 < _analysis("rasch_main")["marginal_reliability"] < 1


def test_invalid_inputs_are_rejected():
    df = _data("rasch_main").copy()
    with pytest.raises(ValueError, match="0/1"):
        R.fit_rasch(df.replace({1.: 2.}))
    const = df.copy(); const["M1"] = 1.
    with pytest.raises(ValueError, match="M1"):
        R.fit_rasch(const)
    with pytest.raises(ValueError, match="en az"):
        R.fit_rasch(df.iloc[:20])
    empty = df.copy(); empty.iloc[0] = np.nan
    assert R.fit_rasch(empty)["dropped_empty_persons"] == 1


def test_standard_errors_refused_when_not_converged():
    f = R.fit_rasch(_data("rasch_main"), max_iter=2)
    assert not f["converged"] and np.isnan(f["items"]["SH(b)"]).all() and f["se_reason"]
