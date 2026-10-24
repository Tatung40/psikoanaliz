"""Parametric-bootstrap LR test for 3PL/4PL DIF of ONE studied item with predeclared anchors.

Why: in 3PL/4PL DIF the nested models frequently put c (or d) on its bound, where the chi-square
reference of the LR statistic is invalid (Self & Liang, 1987); core.irt_dif34 therefore withholds the
asymptotic p-value. Here the null distribution is approximated by simulation from the fitted null model
(studied item invariant, focal mean/SD free), refitting BOTH nested models on every replicate with the
same bounded MML estimator (core.irt_mg34).
    p = (1 + #{T*_b >= T_obs}) / (B_ok + 1)
Only replicates where both nested fits converged and logL_alt >= logL_null - 1e-6 count. If more than
`max_fail_rate` of the requested replicates fail, NO p-value is reported (an incomplete bootstrap is not
a result). Known limitation: bootstrap consistency is not guaranteed when the true null parameter lies
exactly on the boundary (Andrews, 2000); the size of this procedure was therefore checked by the
pre-registered simulation in validation/dif34_bootstrap_size_study.py; its "ui_enabled" result decides
whether ui/dif34_bootstrap_view.py may show the p-value for a model.
"""
import numpy as np
import pandas as pd

from core.irt_dif import _prepare
from core.irt_mg34 import Design34, default_bounds, fit as fit34, irf

NESTED_TOL = 1e-6


def _types(k, item, anchors, studied_role):
    t = ["all"] * k
    for j in anchors:
        t[j] = "anchor"
    t[item] = studied_role
    return t


def _lr(x, obs, groups, model, item, anchors, bounds, points, start0=None, start1=None):
    k = x.shape[1]
    d0 = Design34(model, _types(k, item, anchors, "anchor"), 2, bounds)
    d1 = Design34(model, _types(k, item, anchors, "all"), 2, bounds)
    f0 = fit34(x, obs, groups, d0, "gauss-hermite", points, start=start0, information=False)
    f1 = fit34(x, obs, groups, d1, "gauss-hermite", points, start=start1, information=False)
    ok = bool(f0["converged"] and f1["converged"] and np.isfinite(f0["loglik"]) and np.isfinite(f1["loglik"])
              and f1["loglik"] >= f0["loglik"] - NESTED_TOL)
    return max(2 * (f1["loglik"] - f0["loglik"]), 0.), ok, f0, f1


def _simulate(f0, groups, obs, rng):
    d = f0["design"]; x = np.zeros(obs.shape)
    for g in (0, 1):
        ix = np.flatnonzero(groups == g)
        mu, sd = d.latent_of(f0["params"], g)
        th = mu + sd * rng.standard_normal(len(ix))
        a, b, c, dd = d.unpack(f0["params"], g)
        if d.model == "3PL":
            dd = np.ones_like(a)
        P, _ = irf(th, a, b, c, dd)
        x[ix] = (rng.random(P.shape) < P).astype(float)
    return np.where(obs, x, 0.)


def bootstrap_dif_test(data, group, reference, focal, item, anchors, *, model="3PL", B=199, seed=2026,
                       points=41, bounds=None, max_fail_rate=.10):
    """Return dict with T_obs, p (or NaN), replicate accounting and the fitted null/alternative."""
    if model not in ("3PL", "4PL"):
        raise ValueError("Bootstrap DIF testi yalnızca 3PL/4PL içindir.")
    if not isinstance(B, (int, np.integer)) or B < 19:
        raise ValueError("Bootstrap tekrar sayısı en az 19 olmalıdır.")
    x, obs, groups, _ = _prepare(data, group, reference, focal)
    names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"M{i + 1}" for i in range(x.shape[1])]
    item_ix = names.index(item) if isinstance(item, str) else int(item)
    anc = [names.index(a) if isinstance(a, str) else int(a) for a in anchors]
    if item_ix in anc or len(set(anc)) != len(anc) or len(anc) < 2:
        raise ValueError("En az iki benzersiz çapa gerekir ve incelenen madde çapa olamaz.")
    bounds = bounds or default_bounds()
    T, ok, f0, f1 = _lr(x, obs, groups, model, item_ix, anc, bounds, points)
    out = {"item": names[item_ix], "anchors": [names[j] for j in anc], "model": model, "T_obs": T,
           "observed_valid": ok, "B_requested": int(B), "seed": int(seed), "p": np.nan,
           "boundary_null": bool(f0["at_bound"].any()), "boundary_alt": bool(f1["at_bound"].any())}
    if not ok:
        out.update(B_ok=0, B_failed=0, status="Gözlenen modeller yakınsamadı veya iç içe olabilirlik ihlali; "
                                              "bootstrap yapılmadı.")
        return out
    rng = np.random.default_rng(seed); stats = []; failed = 0
    for _ in range(B):
        xs = _simulate(f0, groups, obs, rng)
        try:
            t, good, _, _ = _lr(xs, obs, groups, model, item_ix, anc, bounds, points,
                                start0=f0["params"], start1=f1["params"])
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            good = False
        if good:
            stats.append(t)
        else:
            failed += 1
    stats = np.array(stats)
    out.update(B_ok=int(len(stats)), B_failed=int(failed), null_stats=stats.tolist())
    if failed > max_fail_rate * B:
        out["status"] = (f"Bootstrap tamamlanamadı: {failed}/{B} tekrar başarısız (> %{100 * max_fail_rate:.0f}); "
                         "p değeri raporlanmadı.")
        return out
    out["p"] = float((1 + (stats >= T - 1e-12).sum()) / (len(stats) + 1))
    out["status"] = "Tamamlandı"
    return out
