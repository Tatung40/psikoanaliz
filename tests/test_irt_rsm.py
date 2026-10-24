"""Rating Scale Model: algebra, analytic score, library cross-check, SEs, persons, fit and LR test."""
import numpy as np
import pandas as pd
import pytest

from core import irt_rsm as R

DELTA = np.linspace(-1.5, 1.5, 8)
TAU = np.array([-1.2, -.3, .4, 1.1])


@pytest.fixture(scope="module")
def data():
    return pd.DataFrame(R.simulate(DELTA, TAU, 1500, 1.2, seed=11), columns=[f"I{j}" for j in range(8)])


@pytest.fixture(scope="module")
def fitted(data):
    return R.fit(data)


def test_probabilities_sum_to_one_and_match_closed_form():
    th = np.linspace(-4, 4, 9); P = np.exp(R.category_log_probs(th, R.rsm_steps(DELTA, TAU)))
    assert np.allclose(P.sum(2), 1, atol=1e-13)
    j, t = 2, 0.7
    num = np.exp([sum(t - DELTA[j] - TAU[s] for s in range(c)) for c in range(5)])
    direct = np.exp(R.category_log_probs(np.array([t]), R.rsm_steps(DELTA, TAU)))[0, j]
    assert np.allclose(direct, num / num.sum(), atol=1e-14)


def test_information_equals_derivative_of_expected_score():
    th = np.linspace(-3, 3, 13); steps = R.rsm_steps(DELTA, TAU); h = 1e-5
    E = lambda t: R.item_moments(t, steps)[1]
    assert np.allclose(R.information(th, steps), (E(th + h) - E(th - h)) / (2 * h), atol=1e-7)


@pytest.mark.parametrize("model", ["RSM", "PCM"])
def test_analytic_score_matches_finite_differences(data, model):
    prep = R.prepare(data); C = prep["C"]; spec = R._Spec(model, 8, C, "rasch")
    x, obs = prep["x"], prep["obs"]
    onehot = ((np.arange(C)[None, None, :] == x[:, :, None]) & obs[:, :, None]).astype(float)
    ge = np.stack([((x >= s) & obs).sum(0) for s in range(1, C)], 1).astype(float)
    z, lw = R._nodes(31); args = (onehot, obs.astype(float), ge, x.sum(1).astype(float), z, lw, len(x))
    v = R._start(prep, spec) + .07
    g = R._negloglik(v, spec, *args)[1]
    num = np.array([(R._negloglik(v + e * 1e-6, spec, *args)[0] - R._negloglik(v - e * 1e-6, spec, *args)[0]) / 2e-6
                    for e in np.eye(len(v))])
    assert np.abs(num - g).max() < 1e-7


def test_fit_recovers_parameters_and_reports_valid_ses(fitted):
    assert fitted["converged"] and fitted["max_score_per_person"] < 1e-6
    assert np.abs(fitted["delta"] - DELTA).max() < .2 and np.abs(fitted["tau"] - TAU).max() < .15
    assert abs(fitted["sd"] - 1.2) < .1 and np.isclose(fitted["tau"].sum(), 0, atol=1e-12)
    t = fitted["tables"]
    assert t["items"]["SH(δ)"].between(.01, .2).all() and t["steps"]["SH(τ)"].between(.005, .2).all()
    assert t["steps"]["Not"].iloc[-1].startswith("türetilmiş")
    assert np.isfinite(t["latent"]["SH(σ)"]).all()


def test_derived_last_step_se_by_delta_method(fitted):
    k = fitted["k"]; sub = fitted["cov"][k:k + 3, k:k + 3]
    assert np.isclose(fitted["tables"]["steps"]["SH(τ)"].iloc[-1], np.sqrt(np.ones(3) @ sub @ np.ones(3)))


def test_unit_variance_fit_equals_python_mirt_rating_scale_model():
    mirt = pytest.importorskip("mirt")
    x = R.simulate(DELTA, TAU, 1000, 1.0, seed=12)
    f = R.fit(pd.DataFrame(x), identification="unit_variance", points=61)
    m = mirt.RatingScaleModel(n_items=8, n_categories=5)
    res = mirt.EMEstimator(n_quadpts=61, tol=1e-8, max_iter=5000).fit(m, x)
    mp = R.to_mirt_parameterization(f["delta"], f["tau"])
    assert abs(f["loglik"] - res.log_likelihood) < 1e-4
    assert np.abs(mp["difficulty"] - m._parameters["difficulty"]).max() < 1e-3
    assert np.abs(mp["thresholds"] - m._parameters["thresholds"]).max() < 1e-3
    th = np.linspace(-3, 3, 7)
    assert np.allclose(np.asarray(m.probability(th[:, None])), np.exp(R.category_log_probs(th, f["steps"])), atol=1e-4)


def test_persons_wle_eap_and_extremes(fitted):
    p = R.persons(fitted)
    assert len(p) == fitted["n"] and np.isfinite(p["WLE θ"]).all() and np.isfinite(p["EAP θ"]).all()
    ext = p["Uç puan"].to_numpy()
    if ext.any():
        assert np.abs(p.loc[ext, "WLE θ"]).max() < 10
    order = np.argsort(p["Ham puan"].to_numpy())
    assert np.all(np.diff(p["WLE θ"].to_numpy()[order]) >= -1e-8)  # complete data: monotone in raw score


def test_item_fit_near_one_with_true_theta():
    rng = np.random.default_rng(13); th = 1.0 * rng.standard_normal(4000)
    P = np.exp(R.category_log_probs(th, R.rsm_steps(DELTA, TAU)))
    x = (rng.random((4000, 8))[..., None] > np.cumsum(P, 2)).sum(2)
    f = R.fit(pd.DataFrame(x))
    fit_true = R.item_fit(f, theta=th)
    assert fit_true["Infit MNSQ"].between(.93, 1.07).all() and fit_true["Outfit MNSQ"].between(.9, 1.1).all()


def test_rsm_vs_pcm_lr_and_input_guards(data):
    lr = R.rsm_vs_pcm(data)
    assert lr["valid"] and lr["df"] == 8 * 4 - (8 + 3) and 0 <= lr["p"] <= 1
    assert lr["pcm"]["loglik"] >= lr["rsm"]["loglik"] - 1e-6
    with pytest.raises(ValueError, match="en az 3"):
        R.fit(pd.DataFrame(np.random.default_rng(0).integers(0, 2, (100, 5))))
    gap = data.replace(2, 3)
    with pytest.raises(ValueError, match="gözlenmeyen"):
        R.fit(gap)
    with pytest.raises(ValueError):
        R.fit(data.astype(str).replace("1", "x"))


def test_missing_responses_are_skipped(data):
    d = data.astype(float).copy(); d.iloc[::7, 2] = np.nan
    f = R.fit(d)
    assert f["converged"] and f["missing"] == int(d.isna().sum().sum())
