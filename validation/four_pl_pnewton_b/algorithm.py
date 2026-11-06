"""Bertsekas (1982) projected Newton for MAXIMIZING f on a box [lo, hi].
info(x) = -Hessian (positive definite at a regular maximum); g = gradient; ascent direction d_F = info_FF^{-1} g_F.
eps-active set: x_i - lo_i <= eps with g_i < 0, or hi_i - x_i <= eps with g_i > 0; eps = min(eps0, ||x - P(x + g)||).
Active coordinates move by the diagonally scaled gradient; the step is the projected arc x(t) = P(x + t d), never a
clipped full step accepted without test. Armijo on the arc: f(x(t)) - f(x) >= sigma [t g_F.d_F + g_A.(x(t) - x)_A].
"""
import time
import numpy as np


def projected_score(x, g, lo, hi):
    return np.where(x <= lo, np.maximum(g, 0), np.where(x >= hi, np.minimum(g, 0), g))


def maximize(f, grad, info, x0, lo, hi, tol=1e-5, eps0=1e-3, sigma=1e-4, max_back=20, max_steps=30, budget=900.,
             cond_max=1e8, log=None):
    x = np.clip(np.asarray(x0, float), lo, hi); fx = f(x); t0 = time.monotonic(); history = []
    for step in range(max_steps + 1):
        g = grad(x); ps = float(np.abs(projected_score(x, g, lo, hi)).max())
        rec = dict(step=step, elapsed=time.monotonic() - t0, score=ps, f=fx,
                   at_bound=int(np.sum((x <= lo) | (x >= hi)))); history.append(rec)
        if log: log(rec)
        if ps <= tol: rec["stop"] = "STATIONARY"; break
        if rec["elapsed"] > budget: rec["stop"] = "BUDGET"; break
        if step == max_steps: rec["stop"] = "MAX STEPS"; break
        eps = min(eps0, float(np.linalg.norm(x - np.clip(x + g, lo, hi))))
        A = ((x - lo <= eps) & (g < 0)) | ((hi - x <= eps) & (g > 0)); F = ~A
        H = info(x); H = (H + H.T) / 2; HF = H[np.ix_(F, F)]; ev = np.linalg.eigvalsh(HF)
        rec.update(active=int(A.sum()), info_min=float(ev[0]), condition=float(ev[-1] / ev[0]) if ev[0] > 0 else float("inf"))
        if ev[0] <= 0 or rec["condition"] > cond_max: rec["stop"] = "INFORMATION GATE"; break
        d = np.zeros_like(x); d[F] = np.linalg.solve(HF, g[F]); d[A] = g[A] / np.maximum(np.diag(H)[A], 1e-12)
        rec["ascent_gFdF"] = float(g[F] @ d[F]); ok = False
        for k in range(max_back + 1):
            t = .5 ** k; xn = np.clip(x + t * d, lo, hi); fn = f(xn)
            if np.isfinite(fn) and fn - fx >= sigma * (t * g[F] @ d[F] + g[A] @ (xn - x)[A]):
                rec.update(t=t, step_inf=float(np.abs(xn - x).max()), dF=float(fn - fx)); x, fx, ok = xn, fn, True; break
        if not ok: rec["stop"] = "STAGNATION"; break
    return x, fx, history
