"""B8: core.rasch against R mirt 1.48 (same MML conditions) and RaschPy 1.4.0 (PAIR).

Fixtures: tests/fixtures/rasch/mirt_rasch_results.csv (scripts/mirt_rasch_reference.R)
and raschpy_results.csv (scripts/raschpy_reference.py); versions in *_session.txt.

mirt: Rasch = 2PL with equal slopes and N(0,1) on the same GH-41 grid (a = sigma,
d = -b); compared for logLik, estimates, SEs (via mirt's logLik Hessian), WLE and
Infit/Outfit. mirt's ZSTD differs from the Wright-Masters formula used here by three
conventions found in its source (q truncated at sqrt 2; W floored when W^2 < 1e-5;
person q.outfit uses all items J in the first term and observed items in the second)
plus, with missing data, an R vector-recycling defect in item q.outfit. An emulation
of exactly these reproduces mirt to ~1e-13, so no difference is left unexplained.

RaschPy: item locations are PAIR estimates (zero-centred), NOT MML; they are compared
only as agreement indicators after centring. Formula checks use RaschPy's own
estimates as inputs. RaschPy's item Outfit ZSTD uses n including extreme persons in
q while the MNSQ excludes them; it is reproduced only under that convention.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
import core.rasch as R

FIX = Path(__file__).parent / "fixtures" / "rasch"
CASES = ["rasch_main", "rasch_missing"]
FIT = ["Infit MNSQ", "Infit ZSTD", "Outfit MNSQ", "Outfit ZSTD"]
MIRT_NAME = {"Infit MNSQ": "infit", "Infit ZSTD": "z.infit", "Outfit MNSQ": "outfit", "Outfit ZSTD": "z.outfit"}


@lru_cache(maxsize=None)
def _data(case):
    return pd.read_csv(FIX / f"data_{case}.csv", na_values="NA")


@lru_cache(maxsize=None)
def _ours(case):
    return R.rasch_analysis(_data(case))


@lru_cache(maxsize=None)
def _mirt():
    return pd.read_csv(FIX / "mirt_rasch_results.csv", keep_default_na=False, na_values=["NA"])


@lru_cache(maxsize=None)
def _raschpy():
    return pd.read_csv(FIX / "raschpy_results.csv", keep_default_na=False, na_values=["NA"])


def _q(table, case, quantity):
    t = table[(table.case == case) & (table.quantity == quantity)]
    return t


def _persons(case):
    return [str(i + 1) for i in range(len(_data(case)))]


def _mirt_params(case):
    cols = list(_data(case).columns)
    a = float(_q(_mirt(), case, "slope").value.iloc[0])
    b = -_q(_mirt(), case, "intercept").set_index("row").value.loc[cols].to_numpy(float)
    w = _q(_mirt(), case, "wle")
    z = w[w.col == "z"].set_index("row").value.loc[_persons(case)].to_numpy(float)
    return a, b, a * z                                 # WLE is invariant to theta = a z


def _extreme(case):
    x, obs, _, _ = R.validate_rasch_data(_data(case))
    raw = np.where(obs, x, 0.).sum(1)
    return x, obs, (raw == 0) | (raw == obs.sum(1))


# --------------------------------------------------------------------------- mirt
def test_mirt_setup_recorded():
    for case in CASES:
        assert _q(_mirt(), case, "prior_types").row.iloc[0] == "none"
        assert _q(_mirt(), case, "converged").value.iloc[0] == 1
    assert "mirt 1.48" in (FIX / "mirt_rasch_session.txt").read_text()


@pytest.mark.parametrize("case", CASES)
def test_mirt_fixed_parameter_loglik(case):
    a, b, _ = _mirt_params(case)
    expected = float(_q(_mirt(), case, "loglik").value.iloc[0])
    assert R.marginal_loglik(_data(case), b, a, 41) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("case", CASES)
def test_mirt_free_mml_estimates(case):
    a, b, _ = _mirt_params(case)
    f = _ours(case)["fit"]
    assert np.abs(f["b"] - b).max() <= 1e-6 and abs(f["sd"] - a) <= 1e-6
    assert f["loglik"] >= float(_q(_mirt(), case, "loglik").value.iloc[0]) - 1e-8


@pytest.mark.parametrize("case", CASES)
def test_mirt_standard_errors(case):
    a, _, _ = _mirt_params(case)
    cols = list(_data(case).columns); k = len(cols)
    names = ["a"] + [f"d.{c}" for c in cols]
    h = _q(_mirt(), case, "hessian").pivot(index="row", columns="col", values="value").loc[names, names].to_numpy()
    g = _q(_mirt(), case, "gradient").set_index("row").value.loc[names].to_numpy()
    jac = np.zeros((k + 1, k + 1)); jac[0, k] = a; jac[1:, :k] = -np.eye(k)   # (b, log sd) -> (a, d)
    hp = jac.T @ h @ jac; hp[k, k] += g[0] * a                                  # second-order term
    cov = np.linalg.inv(-hp)
    f = _ours(case)["fit"]
    np.testing.assert_allclose(f["items"]["SH(b)"], np.sqrt(np.diag(cov)[:k]), rtol=1e-6)
    assert f["se_sd"] == pytest.approx(a * np.sqrt(cov[k, k]), rel=1e-6)


@pytest.mark.parametrize("case", CASES)
def test_mirt_wle(case):
    a, b, theta_mirt = _mirt_params(case)
    x, obs, _ = _extreme(case)
    theta, _ = R.wle(x, obs, b)
    assert np.abs(theta - theta_mirt).max() <= 1e-5          # mirt root-finding tolerance (~3e-6)


@pytest.mark.parametrize("case", CASES)
def test_mirt_mnsq_equal(case):
    _, b, theta = _mirt_params(case)
    x, obs, ext = _extreme(case)
    item, person = R.fit_statistics(x, obs, theta, b)        # mirt uses all persons
    cols = list(_data(case).columns)
    for name in ("Infit MNSQ", "Outfit MNSQ"):
        mi = _q(_mirt(), case, "itemfit"); mi = mi[mi.col == MIRT_NAME[name]].set_index("row").value.loc[cols]
        mp = _q(_mirt(), case, "personfit"); mp = mp[mp.col == MIRT_NAME[name]].set_index("row").value.loc[_persons(case)]
        np.testing.assert_allclose(item[name], mi, atol=1e-4)             # W floor only affects extremes
        np.testing.assert_allclose(person[name][~ext], mp.to_numpy()[~ext], atol=1e-10)


def _mirt_convention_fit(x, obs, theta, b):
    """Emulation of mirt 1.48 itemfit/personfit infit-outfit code (for testing only)."""
    zs = lambda ms, q: (np.cbrt(ms) - 1) * (3 / q) + q / 3
    p = expit(theta[:, None] - b); w = p * (1 - p); c = p * (1 - p) ** 4 + (1 - p) * p ** 4
    w = np.where(obs, np.where(w ** 2 < 1e-5, np.sqrt(1e-5), w), np.nan)
    c0 = np.where(obs, c, 0.); res = np.where(obs, np.nan_to_num(x) - p, 0.)
    n = obs.sum(0); rows, j = obs.shape
    recycled = n[(np.arange(rows)[:, None] + np.arange(j)[None, :] * rows) % j]   # R recycling
    io = np.nansum(res ** 2 / w, 0) / n
    qo = np.minimum(np.sqrt(np.nansum(c0 / w ** 2 / recycled ** 2, 0) - 1 / n), 1.4142)
    ii = np.nansum(res ** 2, 0) / np.nansum(w, 0)
    qi = np.minimum(np.sqrt(np.nansum(np.where(obs, c, np.nan) - w ** 2, 0) / np.nansum(w, 0) ** 2), 1.4142)
    k = obs.sum(1)
    po = np.nansum(res ** 2 / w, 1) / k
    pqo = np.minimum(np.sqrt(np.abs(np.nansum(np.where(obs, c, np.nan) / w ** 2 / j ** 2, 1) - 1 / k)), 1.4142)
    pi = np.nansum(res ** 2, 1) / np.nansum(w, 1)
    pqi = np.minimum(np.sqrt(np.nansum(np.where(obs, c, np.nan) - w ** 2, 1) / np.nansum(w, 1) ** 2), 1.4142)
    return ({"Outfit ZSTD": zs(io, qo), "Infit ZSTD": zs(ii, qi)},
            {"Outfit ZSTD": zs(po, pqo), "Infit ZSTD": zs(pi, pqi)})


@pytest.mark.parametrize("case", CASES)
def test_mirt_zstd_differences_fully_explained(case):
    _, b, theta = _mirt_params(case)
    x, obs, ext = _extreme(case)
    emu_item, emu_person = _mirt_convention_fit(x, obs, theta, b)
    ours_item, ours_person = R.fit_statistics(x, obs, theta, b)
    cols = list(_data(case).columns)
    for name in ("Infit ZSTD", "Outfit ZSTD"):
        mi = _q(_mirt(), case, "itemfit"); mi = mi[mi.col == MIRT_NAME[name]].set_index("row").value.loc[cols].to_numpy()
        mp = _q(_mirt(), case, "personfit"); mp = mp[mp.col == MIRT_NAME[name]].set_index("row").value.loc[_persons(case)].to_numpy()
        assert np.abs(emu_item[name] - mi).max() <= 1e-9
        assert np.nanmax(np.abs(emu_person[name] - mp)) <= 1e-9
    # and the conventions matter: our uncapped, consistent formula differs where they bite
    mp = _q(_mirt(), case, "personfit"); mp = mp[mp.col == "z.outfit"].set_index("row").value.loc[_persons(case)].to_numpy()
    assert np.nanmax(np.abs(ours_person["Outfit ZSTD"] - mp)) > .1


def test_mirt_missing_data_item_outfit_zstd_recycling_defect():
    case = "rasch_missing"
    _, b, theta = _mirt_params(case)
    x, obs, _ = _extreme(case)
    assert len(set(obs.sum(0))) > 1                                  # unequal item counts
    ours, _ = R.fit_statistics(x, obs, theta, b)
    mi = _q(_mirt(), case, "itemfit"); mi = mi[mi.col == "z.outfit"].set_index("row").value.loc[list(_data(case).columns)].to_numpy()
    assert np.abs(ours["Outfit ZSTD"] - mi).max() > .01               # reported, not hidden


# ------------------------------------------------------------------------ RaschPy
def _raschpy_inputs(case):
    cols = list(_data(case).columns)
    b = _q(_raschpy(), case, "item_location").set_index("row").value.loc[cols].to_numpy(float)
    theta = _q(_raschpy(), case, "person_wle").set_index("row").value.loc[_persons(case)].to_numpy(float)
    return b, theta


def test_raschpy_version_recorded():
    assert "RaschPy 1.4.0" in (FIX / "raschpy_session.txt").read_text()


@pytest.mark.parametrize("case", CASES)
def test_raschpy_wle_given_its_items(case):
    b, theta_r = _raschpy_inputs(case)
    x, obs, _ = _extreme(case)
    theta, _ = R.wle(x, obs, b)
    assert np.abs(theta - theta_r).max() <= 1e-9


@pytest.mark.parametrize("case", CASES)
def test_raschpy_fit_formulas_given_its_estimates(case):
    b, theta = _raschpy_inputs(case)
    x, obs, ext = _extreme(case)
    item, person = R.fit_statistics(x, obs, theta, b, use=~ext)
    cols = list(_data(case).columns)
    for name in FIT:
        rp = _q(_raschpy(), case, "person_fit"); rp = rp[rp.col == name].set_index("row").value
        ours = pd.Series(person[name], index=_persons(case)).loc[rp.index]
        np.testing.assert_allclose(ours, rp, atol=1e-9)
        if name != "Outfit ZSTD":
            ri = _q(_raschpy(), case, "item_fit"); ri = ri[ri.col == name].set_index("row").value.loc[cols]
            np.testing.assert_allclose(item[name], ri, atol=1e-9)


@pytest.mark.parametrize("case", CASES)
def test_raschpy_item_outfit_zstd_uses_inconsistent_n(case):
    b, theta = _raschpy_inputs(case)
    x, obs, ext = _extreme(case)
    item, _ = R.fit_statistics(x, obs, theta, b, use=~ext)
    ri = _q(_raschpy(), case, "item_fit"); ri = ri[ri.col == "Outfit ZSTD"].set_index("row").value.loc[list(_data(case).columns)].to_numpy()
    assert np.abs(item["Outfit ZSTD"] - ri).max() > .05                # consistent n (ours) differs
    p = expit(theta[:, None] - b); w = p * (1 - p); c = p * (1 - p) ** 4 + (1 - p) * p ** 4
    m = obs & ~ext[:, None]
    n_all = obs.sum(0)                                                  # RaschPy: extremes in n only
    q = np.sqrt(np.where(m, c / w ** 2, 0.).sum(0) / n_all ** 2 - 1 / n_all)
    assert np.abs((np.cbrt(item["Outfit MNSQ"]) - 1) * 3 / q + q / 3 - ri).max() <= 1e-9


@pytest.mark.parametrize("case", CASES)
def test_pair_and_mml_are_close_but_not_equivalent(case):
    b_pair, theta_pair = _raschpy_inputs(case)
    b_mml = _ours(case)["fit"]["b"]
    centred = b_mml - b_mml.mean()
    assert abs(b_pair.mean()) <= 1e-8                                   # PAIR is item-centred
    assert np.corrcoef(centred, b_pair)[0, 1] >= .998
    assert 1e-3 < np.abs(centred - b_pair).max() < .25                  # different estimators
    assert np.corrcoef(_ours(case)["persons"]["WLE θ"], theta_pair)[0, 1] >= .9999


def test_live_raschpy_rebuild_reproduces_fixture():
    raschpy = pytest.importorskip("raschpy")
    import warnings
    case = "rasch_missing"
    data = _data(case).copy(); data.index = _persons(case)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = raschpy.SLM(data); model.calibrate()                      # fresh object
    b, _ = _raschpy_inputs(case)
    np.testing.assert_allclose(model.items.loc[list(data.columns)].to_numpy(), b, atol=1e-9)
