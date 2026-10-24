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
