"""Shared, independent pieces for the binary certification protocol (PROTOCOL.json).

Truths, data generation, and an independent numpy objective (marginal log-likelihood with Gauss-Hermite quadrature,
plus the library default log-priors for MAP) with a numerical observed-information Hessian. Nothing here calls
Python mirt; it is the arbiter for G1 and the independent SE reference for MAP models.
"""
import numpy as np
from scipy.special import expit, roots_hermite
from scipy.stats import beta, lognorm, norm

PRIORS = {"a": lognorm(0.5), "b": norm(0, 2), "c": beta(2, 8), "d": beta(8, 2)}


def truth(model, k, variant=None):
    b = np.linspace(-1.5, 1.5, k)
    if model == "1PL":
        return {"a": np.full(k, 1.3 if variant == "common_a" else 1.0), "b": b}
    if model == "2PL":
        return {"a": np.linspace(0.8, 2.0, k), "b": b}
    from validation.irt_3pl4pl_root_cause import truth as t34
    return t34(model, k)


def simulate(model, n, k, seed, variant=None):
    rng = np.random.default_rng(seed)
    p = truth(model, k, variant)
    theta = rng.standard_normal(n)
    c = p.get("c", 0.0); d = p.get("d", 1.0) if model == "4PL" else 1.0
    prob = c + (d - c) * expit(p["a"] * (theta[:, None] - p["b"]))
    return (rng.random(prob.shape) < prob).astype(int)


class Objective:
    """log L (+ log prior) of the binary model at IRT parameters; free-parameter vector layout given by `free`."""

    def __init__(self, x, model, map_priors=False, points=121):
        z, w = roots_hermite(points)
        self.theta, self.w = np.sqrt(2) * z, w / np.sqrt(np.pi)
        self.x, self.model, self.map = np.asarray(x, float), model, map_priors

    def value(self, p):
        k = self.x.shape[1]
        a, b = np.broadcast_to(p["a"], k), np.asarray(p["b"], float)
        c = np.asarray(p.get("c", np.zeros(k)), float); d = np.asarray(p.get("d", np.ones(k)), float)
        prob = np.clip(c + (d - c) * expit(a * (self.theta[:, None] - b)), 1e-300, 1 - 1e-16)
        L = self.x @ np.log(prob).T + (1 - self.x) @ np.log1p(-prob).T
        m = L.max(1, keepdims=True)
        out = float(np.sum(m[:, 0] + np.log(np.exp(L - m) @ self.w)))
        if self.map:
            out += sum(float(np.sum(PRIORS[q].logpdf(np.atleast_1d(p[q])))) for q in p if q in PRIORS)
        return out

    def se(self, p, free, h=1e-4):
        """SE from the numerical inverse Hessian; free: list of (name, index or None for a shared scalar)."""
        def unpack(v):
            q = {key: np.array(val, float, copy=True) for key, val in p.items()}
            for value, (name, j) in zip(v, free):
                if j is None:
                    q[name] = np.full_like(q[name], value)
                else:
                    q[name][j] = value
            return q
        v0 = np.array([np.atleast_1d(p[name])[0 if j is None else j] for name, j in free], float)
        f = lambda v: self.value(unpack(v))
        m = len(v0); H = np.zeros((m, m)); E = np.eye(m) * h
        for i in range(m):
            for j in range(i, m):
                H[i, j] = H[j, i] = (f(v0 + E[i] + E[j]) - f(v0 + E[i] - E[j]) - f(v0 - E[i] + E[j])
                                     + f(v0 - E[i] - E[j])) / (4 * h * h)
        return np.sqrt(np.diag(np.linalg.inv(-H)))
