"""Independent numerical reference for 3PL/4PL MAP (no Python mirt, no R): marginal log-posterior with Gauss-Hermite
quadrature, exact gradient by Fisher's identity (E-step expected counts x analytic item derivatives), L-BFGS-B within
the Python mirt library box, from a data-only start (a = 1, b = -logit(p), c = .15, d = .95)."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, roots_hermite

PRIOR = {"a": (lambda a: -np.log(a) - np.log(a) ** 2 / 0.5, lambda a: -1 / a - np.log(a) / (0.25 * a)),
         "b": (lambda b: -b ** 2 / 8, lambda b: -b / 4),
         "c": (lambda c: np.log(c) + 7 * np.log1p(-c), lambda c: 1 / c - 7 / (1 - c)),
         "d": (lambda d: 7 * np.log(d) + np.log1p(-d), lambda d: 7 / d - 1 / (1 - d))}   # up to constants
BOX = {"a": (0.1, 5.0), "b": (-6.0, 6.0), "c": (1e-6, 0.5), "d": (0.5, 1 - 1e-6)}


def objective(x, model, points=121):
    """(f_g, qs, k): f_g(v) -> (negative log-posterior [log-likelihood for fixed-a 1PL], exact gradient)."""
    x = np.asarray(x, float); n, k = x.shape
    qs = {"1PL": "b", "3PL": "abc", "4PL": "abcd"}[model]; prior = model != "1PL"
    z, w = roots_hermite(points); th = np.sqrt(2) * z; lw = np.log(w / np.sqrt(np.pi))
    def f_g(v):
        p = {q: v[i::len(qs)] for i, q in enumerate(qs)}
        a = p.get("a", np.ones(k)); c = p.get("c", np.zeros(k)); d = p.get("d", np.ones(k))
        s = expit(a * (th[:, None] - p["b"])); P = np.clip(c + (d - c) * s, 1e-300, 1 - 1e-16)
        L = x @ np.log(P).T + (1 - x) @ np.log1p(-P).T + lw
        m = L.max(1, keepdims=True); post = np.exp(L - m); tot = post.sum(1, keepdims=True); post /= tot
        val = float(np.sum(m[:, 0] + np.log(tot[:, 0])))
        r1 = post.T @ x; r0 = post.T @ (1 - x)
        g = r1 / P - r0 / (1 - P); u = s * (1 - s); t = th[:, None] - p["b"]
        D = {"a": (d - c) * u * t, "b": -(d - c) * u * a, "c": 1 - s, "d": s}
        grad = {q: (g * D[q]).sum(0) + (PRIOR[q][1](p[q]) if prior else 0) for q in qs}
        if prior:
            val += sum(float(np.sum(PRIOR[q][0](p[q]))) for q in qs)
        return -val, -np.ravel(np.column_stack([grad[q] for q in qs]))
    return f_g, qs, k


def data_start(x):
    pbar = np.clip(np.asarray(x, float).mean(0), .02, .98); k = len(pbar)
    return {"a": np.ones(k), "b": -np.log(pbar / (1 - pbar)), "c": np.full(k, .15), "d": np.full(k, .95)}


def fit(x, model, points=121, start=None):
    f_g, qs, k = objective(x, model, points)
    start = start or data_start(x)
    v0 = np.ravel(np.column_stack([np.asarray(start[q], float) for q in qs]))
    r = minimize(f_g, v0, jac=True, method="L-BFGS-B", bounds=[BOX[q] for _ in range(k) for q in qs],
                 options=dict(maxiter=20000, maxfun=40000, ftol=1e-16, gtol=1e-7, maxcor=50))
    return {q: r.x[i::len(qs)] for i, q in enumerate(qs)}, {"converged": bool(r.success), "message": str(r.message),
                                                           "iterations": int(r.nit), "max_abs_gradient": float(np.abs(r.jac).max()),
                                                           "neg_objective": float(r.fun)}
