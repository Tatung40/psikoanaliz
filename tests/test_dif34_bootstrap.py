"""3PL/4PL parametric-bootstrap DIF LR: accounting, failure handling and input guards."""
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

import core.irt_dif34_bootstrap as boot


def data(dif=0., seed=1, k=6, n=400):
    rng = np.random.default_rng(seed)
    a = np.linspace(1., 1.6, k); b = np.linspace(-1, 1, k); c = np.linspace(.05, .2, k)
    g = np.r_[np.zeros(n, int), np.ones(n, int)]
    th = np.r_[rng.standard_normal(n), -.2 + rng.standard_normal(n)]
    bb = np.tile(b, (2 * n, 1)); bb[g == 1, 1] += dif
    x = (rng.random((2 * n, k)) < c + (1 - c) * expit(a * (th[:, None] - bb))).astype(int)
    return pd.DataFrame(x, columns=[f"M{j}" for j in range(k)]), pd.Series(g)


def test_bootstrap_accounting_and_p_value_range():
    x, g = data()
    r = boot.bootstrap_dif_test(x, g, 0, 1, "M1", ["M0", "M2", "M3", "M4", "M5"], B=19, seed=4)
    assert r["status"] == "Tamamlandı" and r["B_ok"] + r["B_failed"] == 19
    assert 1 / 20 <= r["p"] <= 1 and len(r["null_stats"]) == r["B_ok"]
    assert r["p"] == pytest.approx((1 + sum(t >= r["T_obs"] - 1e-12 for t in r["null_stats"])) / (r["B_ok"] + 1))


def test_large_dif_gets_small_bootstrap_p():
    x, g = data(dif=1.2, seed=2)
    r = boot.bootstrap_dif_test(x, g, 0, 1, "M1", ["M0", "M2", "M3", "M4", "M5"], B=19, seed=5)
    assert r["p"] == pytest.approx(1 / 20)


def test_failed_replicates_withhold_p(monkeypatch):
    x, g = data(seed=3)
    real = boot._lr; calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] > 1 and calls["n"] % 3 == 0:
            raise RuntimeError("simulated optimizer failure")
        return real(*a, **k)
    monkeypatch.setattr(boot, "_lr", flaky)
    r = boot.bootstrap_dif_test(x, g, 0, 1, "M1", ["M0", "M2", "M3", "M4", "M5"], B=19, seed=6)
    assert r["B_failed"] > .1 * 19 and np.isnan(r["p"]) and "tamamlanamadı" in r["status"]


def test_input_guards():
    x, g = data(seed=4)
    with pytest.raises(ValueError):
        boot.bootstrap_dif_test(x, g, 0, 1, "M1", ["M0"], B=19)
    with pytest.raises(ValueError):
        boot.bootstrap_dif_test(x, g, 0, 1, "M1", ["M1", "M2"], B=19)
    with pytest.raises(ValueError):
        boot.bootstrap_dif_test(x, g, 0, 1, "M1", ["M0", "M2"], B=5)
    with pytest.raises(ValueError):
        boot.bootstrap_dif_test(x, g, 0, 1, "M1", ["M0", "M2"], model="2PL", B=19)


def _independent_loglik(x, groups, fitted, points=41):
    """Two-group marginal log-likelihood from the unpacked parameters on an independent GH rule."""
    from numpy.polynomial.hermite_e import hermegauss
    from scipy.special import logsumexp
    z, w = hermegauss(points); logw = np.log(w / w.sum()); d = fitted["design"]; total = 0.
    for g in (0, 1):
        mu, sd = d.latent_of(fitted["params"], g)
        a, b, c, dd = d.unpack(fitted["params"], g)
        dd = np.ones_like(a) if d.model == "3PL" else dd
        p = c + (dd - c) * expit(a * (mu + sd * z[:, None] - b))
        y = x[groups == g]
        total += logsumexp(y @ np.log(p).T + (1 - y) @ np.log1p(-p).T + logw, axis=1).sum()
    return float(total)


def test_lr_constraints_likelihoods_and_seed_determinism():
    """Null: studied item equal across groups; alternative: only it free; anchors equal in both; LR and
    both log-likelihoods agree with an independent evaluation; same seed gives the same bootstrap."""
    x, g = data(dif=.5, seed=7)
    y = x.to_numpy(float); groups = g.to_numpy()
    T, ok, f0, f1 = boot._lr(y, np.ones_like(y, bool), groups, "3PL", 1, [0, 2, 3, 4, 5], boot.default_bounds(), 41)
    assert ok
    for fitted, studied_free in ((f0, False), (f1, True)):
        p0, p1 = (np.array(fitted["design"].unpack(fitted["params"], h)) for h in (0, 1))
        anchors = [0, 2, 3, 4, 5]
        np.testing.assert_array_equal(p0[:, anchors], p1[:, anchors])
        assert (not np.array_equal(p0[:, 1], p1[:, 1])) == studied_free
        assert _independent_loglik(y, groups, fitted) == pytest.approx(fitted["loglik"], abs=1e-6)
    assert T == pytest.approx(2 * (f1["loglik"] - f0["loglik"]), abs=1e-12) and T > 0
    args = (x, g, 0, 1, "M1", ["M0", "M2", "M3", "M4", "M5"])
    r1, r2 = boot.bootstrap_dif_test(*args, B=19, seed=11), boot.bootstrap_dif_test(*args, B=19, seed=11)
    assert r1["null_stats"] == r2["null_stats"] and r1["p"] == r2["p"]
    assert r1["p"] == pytest.approx((1 + sum(t >= r1["T_obs"] - 1e-12 for t in r1["null_stats"])) / (r1["B_ok"] + 1))
