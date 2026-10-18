"""Single- and two-group marginal ML for the 1PL and 2PL models (B10).

Item response function (the parameterization of core.irt):
    P(X=1 | theta) = logistic(a_i (theta - b_i)),   1PL: a_i = 1 for all items.
Identification keeps the 1PL/2PL convention of core.irt.fit_irt: theta ~ N(0, 1) in
the (reference) group. With two groups the focal latent mean mu_F and SD sigma_F are
free; the common metric is fixed by the reference N(0, 1) together with anchor items
whose parameters are equal in both groups. The 1PL keeps a = 1 in both groups, so its
focal SD is identified; it is NOT the B8 Rasch model (latent SD free, core.rasch).

Each item has a constraint type:
    "anchor"  a and b equal across groups
    "b"       b group-specific, a common (2PL) or fixed at 1 (1PL)
    "ab"      a and b group-specific (2PL only)
Missing responses (NaN) are skipped in the likelihood (MAR).

Estimation: direct L-BFGS-B on the marginal log-likelihood with the analytic score.
Convergence is decided by the score norm per person (<= 1e-6) with no parameter on a
bound; observed information = central-difference Jacobian of the analytic score; SEs
are refused when it is not positive definite.

quadrature="gauss-hermite" (production; theta = mu + sigma z on GH nodes) or "mirt"
(61 equally spaced nodes on [-6, 6] with normal weights; used to reproduce R mirt
exactly in validation).
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit, logsumexp
from scipy.stats import norm
from core.rasch import _grid, SCORE_TOLERANCE

A_BOUNDS = (0.05, 8.)
B_BOUNDS = (-8., 8.)
MU_BOUNDS = (-4., 4.)
LOG_SD_BOUNDS = (np.log(.1), np.log(10.))
TYPES = ("anchor", "b", "ab")


def as_matrix(data):
    x = np.asarray(data, float)
    if x.ndim != 2 or x.shape[1] < 3:
        raise ValueError("En az 3 madde gereklidir.")
    obs = np.isfinite(x)
    if not np.isin(x[obs], [0., 1.]).all():
        raise ValueError("Yanıtlar 0/1 (veya eksik) olmalıdır.")
    return np.where(obs, x, 0.), obs


class Design:
    """Index of a_ig and b_ig in the parameter vector (-1: a fixed at 1)."""

    def __init__(self, model, types, groups=2):
        if model not in ("1PL", "2PL"):
            raise ValueError("Model 1PL veya 2PL olmalıdır.")
        types = list(types)
        if any(t not in TYPES for t in types):
            raise ValueError("Geçersiz madde kısıt türü.")
        if model == "1PL" and "ab" in types:
            raise ValueError("1PL'de ayırt edicilik sabittir; 'ab' kısıtı tanımsızdır.")
        if groups == 1 and any(t != "anchor" for t in types):
            raise ValueError("Tek grupta yalnızca 'anchor' kısıtı kullanılabilir.")
        self.model, self.types, self.groups, k = model, types, groups, len(types)
        self.a = -np.ones((groups, k), int); self.b = np.zeros((groups, k), int)
        self.labels = []; pos = 0
        for i, t in enumerate(types):
            if model == "2PL":
                if t == "ab":
                    for g in range(groups):
                        self.a[g, i] = pos; self.labels.append(("a", g, i)); pos += 1
                else:
                    self.a[:, i] = pos; self.labels.append(("a", -1, i)); pos += 1
            if t in ("b", "ab"):
                for g in range(groups):
                    self.b[g, i] = pos; self.labels.append(("b", g, i)); pos += 1
            else:
                self.b[:, i] = pos; self.labels.append(("b", -1, i)); pos += 1
        self.latent = pos
        self.size = pos + (2 if groups == 2 else 0)

    def bounds(self):
        out = [A_BOUNDS if kind == "a" else B_BOUNDS for kind, _, _ in self.labels]
        return out + ([MU_BOUNDS, LOG_SD_BOUNDS] if self.groups == 2 else [])

    def unpack(self, params, g):
        a = np.where(self.a[g] >= 0, params[np.maximum(self.a[g], 0)], 1.)
        return a, params[self.b[g]]

    def latent_of(self, params, g):
        if g == 0:
            return 0., 1.
        return float(params[self.latent]), float(np.exp(params[self.latent + 1]))


class Quadrature:
    def __init__(self, kind="gauss-hermite", points=41):
        self.kind = kind
        if kind == "gauss-hermite":
            self.z, self.lw = _grid(points)
        elif kind == "mirt":
            self.theta = np.linspace(-6., 6., 61)
        else:
            raise ValueError("Bilinmeyen kuadratür.")

    def nodes(self, mu, sd):
        if self.kind == "gauss-hermite":
            return mu + sd * self.z, self.lw
        lw = norm.logpdf(self.theta, mu, sd)
        return self.theta, lw - logsumexp(lw)


def _parts(x, obs, groups, G):
    out = []
    for g in range(G):
        m = groups == g
        out.append((np.where(obs[m], x[m], 0.), np.where(obs[m], 1. - x[m], 0.)))
    return out


def negloglik(params, design, parts, quad, n_total):
    grad = np.zeros_like(params); value = 0.
    for g, (x1, x0) in enumerate(parts):
        mu, sd = design.latent_of(params, g)
        theta, lw = quad.nodes(mu, sd)
        a, b = design.unpack(params, g)
        eta = (theta[:, None] - b[None, :]) * a[None, :]
        lp = x1 @ log_expit(eta).T + x0 @ log_expit(-eta).T + lw
        normc = logsumexp(lp, axis=1); value -= normc.sum()
        post = np.exp(lp - normc[:, None])
        resid = post.T @ x1 - (post.T @ (x1 + x0)) * expit(eta)        # Q x k: d logL / d eta
        ga = (resid * (theta[:, None] - b[None, :])).sum(0)
        gb = -(resid * a[None, :]).sum(0)
        free_a = design.a[g] >= 0
        np.add.at(grad, design.a[g][free_a], -ga[free_a])
        np.add.at(grad, design.b[g], -gb)
        if g == 1:
            d_theta = (resid * a[None, :]).sum(1)
            if quad.kind == "gauss-hermite":
                g_mu, g_ls = d_theta.sum(), (d_theta * sd * quad.z).sum()
            else:
                r_q = post.sum(0); w = np.exp(lw); n_g = r_q.sum()
                u = (theta - mu) / sd ** 2; v = (theta - mu) ** 2 / sd ** 2
                g_mu = (r_q * u).sum() - n_g * (w * u).sum()
                g_ls = (r_q * v).sum() - n_g * (w * v).sum()
            grad[design.latent] -= g_mu; grad[design.latent + 1] -= g_ls
    return value / n_total, grad / n_total


def start_values(x, obs, design, groups):
    p = np.clip(x.sum(0) / np.maximum(obs.sum(0), 1), .02, .98)
    b0 = np.clip(-np.log(p / (1 - p)) * (1.7 if design.model == "2PL" else 1.), -3, 3)
    start = np.zeros(design.size)
    for j, (kind, _, i) in enumerate(design.labels):
        start[j] = 1. if kind == "a" else b0[i]
    return start


def fit(x, obs, groups, design, quadrature="gauss-hermite", points=41, start=None, max_iter=5000,
        information=True):
    groups = np.asarray(groups, int)
    quad = Quadrature(quadrature, points)
    parts = _parts(x, obs, groups, design.groups)
    n = len(x)
    bounds = design.bounds()
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    x0 = start_values(x, obs, design, groups) if start is None else np.clip(np.asarray(start, float), lo, hi)
    res = minimize(negloglik, x0, args=(design, parts, quad, n), jac=True, method="L-BFGS-B",
                   bounds=bounds, options={"maxiter": max_iter, "ftol": 1e-15, "gtol": 1e-12})
    params = np.asarray(res.x, float)
    value, score = negloglik(params, design, parts, quad, n)
    at_bound = (np.abs(params - lo) < 1e-6) | (np.abs(params - hi) < 1e-6)
    max_score = float(np.abs(score[~at_bound]).max()) if (~at_bound).any() else 0.
    converged = bool(np.isfinite(params).all() and max_score <= SCORE_TOLERANCE and not at_bound.any())
    out = {"params": params, "loglik": -value * n, "converged": converged, "at_bound": at_bound,
           "max_score_per_person": max_score, "iterations": int(res.nit), "design": design,
           "quadrature": quadrature, "points": points, "cov": None, "se_reason": "", "n": n}
    if not converged:
        out["se_reason"] = ("Kestirim yakınsamadı." if not at_bound.any()
                            else "Bir veya daha fazla parametre optimizasyon sınırında.")
    elif information:
        out["cov"], out["se_reason"] = covariance(params, design, parts, quad, n)
    if design.groups == 2:
        out["mu_F"], out["sd_F"] = design.latent_of(params, 1)
    return out


def covariance(params, design, parts, quad, n):
    m = len(params); info = np.zeros((m, m))
    for j in range(m):
        h = 1e-5 * max(1., abs(params[j]))
        up, dn = params.copy(), params.copy(); up[j] += h; dn[j] -= h
        info[:, j] = (negloglik(up, design, parts, quad, n)[1] - negloglik(dn, design, parts, quad, n)[1]) * n / (2 * h)
    info = (info + info.T) / 2
    eig = np.linalg.eigvalsh(info)
    if not np.isfinite(eig).all() or eig[0] <= 1e-8 * max(eig[-1], 1.):
        return None, "Gözlenen bilgi matrisi pozitif tanımlı değil (tekil veya kötü koşullu); SH üretilemez."
    return np.linalg.inv(info), ""


def loglik_at(x, obs, groups, design, params, quadrature="mirt", points=41):
    """Marginal log-likelihood at fixed parameters (validation)."""
    parts = _parts(x, obs, np.asarray(groups, int), design.groups)
    return -negloglik(np.asarray(params, float), design, parts, Quadrature(quadrature, points), len(x))[0] * len(x)


def item_table(fitted, names, g=0):
    a, b = fitted["design"].unpack(fitted["params"], g)
    return {"Madde": list(names), "a": a, "b": b}


def fit_single(data, model="2PL", quadrature="gauss-hermite", points=41, information=True):
    """Single-group 1PL/2PL MML with theta ~ N(0, 1) (missing responses allowed)."""
    x, obs = as_matrix(data)
    design = Design(model, ["anchor"] * x.shape[1], groups=1)
    return fit(x, obs, np.zeros(len(x), int), design, quadrature, points, information=information)
