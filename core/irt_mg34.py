"""Single- and two-group bounded marginal ML for the 3PL and 4PL models (B11).

Item response function (the parameterization of core.irt.fit_irt, unchanged):
    P(X=1 | theta) = c_i + (d_i - c_i) logistic(a_i (theta - b_i)),
3PL: d_i = 1 fixed; 4PL: d_i free. theta ~ N(0, 1) in the (reference) group; with two
groups the focal mean mu_F and SD sigma_F are free and anchor items share all
parameters. Box constraints are those of the production calibration
(core.irt: a in [0.15, a_upper], b in [-4, 4], c in [0, 0.35], d in [d_lower, 1]); no prior
or penalty is used, so this is bounded (constrained) ML, not MAP.

Constraint type of an item = the set of its parameters that are group-specific:
"anchor" (none), single parameters "a", "b", "c", "d", or "all" (every free parameter).

Convergence: projected-gradient (KKT) norm per person <= 1e-6 (a parameter on a bound
may have an outward-pointing score). `regular` additionally requires that no parameter
is on a bound. The observed information (central-difference Jacobian of the analytic
score) is computed over the interior parameters with bound-active parameters held
fixed (conditional information, flagged by `cov_conditional`); it is refused when not
positive definite or its condition number exceeds 1e10. Callers decide which
contrasts may be used inferentially (the studied parameters must be interior).
"""
from contextlib import nullcontext
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from core.irt_mg import Quadrature, _parts, as_matrix
from core.rasch import SCORE_TOLERANCE
from core.irt_se_diagnostics import BOUNDARY_TOLERANCE

PARAMS = {"3PL": ("a", "b", "c"), "4PL": ("a", "b", "c", "d")}
TYPES = {"anchor": (), "a": ("a",), "b": ("b",), "c": ("c",), "d": ("d",), "all": None}
P_EPS = 1e-12

try:                                   # small matrix products: multithreaded BLAS only adds overhead
    from threadpoolctl import threadpool_limits

    def _single_thread():
        return threadpool_limits(1)
except ImportError:                    # optional dependency
    def _single_thread():
        return nullcontext()


def default_bounds(a_upper=4.0, d_lower=0.65):
    return {"a": (0.15, float(a_upper)), "b": (-4., 4.), "c": (0., .35), "d": (float(d_lower), 1.)}


class Design34:
    def __init__(self, model, types, groups=2, bounds=None):
        if model not in PARAMS:
            raise ValueError("Bu kestirici yalnızca 3PL ve 4PL içindir.")
        self.model, self.types, self.groups = model, list(types), groups
        self.names = PARAMS[model]
        self.bounds_by = bounds or default_bounds()
        k = len(self.types)
        self.idx = {p: -np.ones((groups, k), int) for p in ("a", "b", "c", "d")}
        self.labels = []; pos = 0
        for i, t in enumerate(self.types):
            if t not in TYPES:
                raise ValueError("Geçersiz madde kısıt türü.")
            free = self.names if TYPES[t] is None else TYPES[t]
            if any(p not in self.names for p in free):
                raise ValueError(f"{model} modelinde '{t}' kısıtı tanımsızdır.")
            if groups == 1 and free:
                raise ValueError("Tek grupta yalnızca 'anchor' kısıtı kullanılabilir.")
            for p in self.names:
                if p in free:
                    for g in range(groups):
                        self.idx[p][g, i] = pos; self.labels.append((p, g, i)); pos += 1
                else:
                    self.idx[p][:, i] = pos; self.labels.append((p, -1, i)); pos += 1
        self.latent = pos
        self.size = pos + (2 if groups == 2 else 0)

    def bounds(self):
        return [self.bounds_by[p] for p, _, _ in self.labels] + (
            [(-4., 4.), (np.log(.1), np.log(10.))] if self.groups == 2 else [])

    def unpack(self, params, g):
        out = {}
        for p, fixed in (("a", None), ("b", None), ("c", 0.), ("d", 1.)):
            ix = self.idx[p][g]
            out[p] = np.where(ix >= 0, params[np.maximum(ix, 0)], fixed if fixed is not None else np.nan)
        return out["a"], out["b"], out["c"], out["d"]

    def latent_of(self, params, g):
        if g == 0:
            return 0., 1.
        return float(params[self.latent]), float(np.exp(params[self.latent + 1]))


def irf(theta, a, b, c, d):
    L = expit(a[None, :] * (np.asarray(theta, float)[:, None] - b[None, :]))
    return c[None, :] + (d - c)[None, :] * L, L


def negloglik(params, design, parts, quad, n_total):
    grad = np.zeros_like(params); value = 0.
    for g, (x1, x0) in enumerate(parts):
        mu, sd = design.latent_of(params, g)
        theta, lw = quad.nodes(mu, sd)
        a, b, c, d = design.unpack(params, g)
        P, L = irf(theta, a, b, c, d)
        P = np.clip(P, P_EPS, 1 - P_EPS)
        lp = x1 @ np.log(P).T + x0 @ np.log1p(-P).T + lw
        normc = logsumexp(lp, axis=1); value -= normc.sum()
        post = np.exp(lp - normc[:, None])
        G = (post.T @ x1 - (post.T @ (x1 + x0)) * P) / (P * (1 - P))        # Q x k: d logL / d P
        dL = (d - c)[None, :] * L * (1 - L)                                   # dP / d eta
        GE = G * dL
        parts_grad = {"a": (GE * (theta[:, None] - b[None, :])).sum(0), "b": -(GE * a[None, :]).sum(0),
                      "c": (G * (1 - L)).sum(0), "d": (G * L).sum(0)}
        for p in design.names:
            np.add.at(grad, design.idx[p][g], -parts_grad[p])
        if g == 1:
            d_theta = (GE * a[None, :]).sum(1)
            if quad.kind == "gauss-hermite":
                g_mu, g_ls = d_theta.sum(), (d_theta * sd * quad.z).sum()
            else:
                r_q = post.sum(0); w = np.exp(lw); n_g = r_q.sum()
                u = (theta - mu) / sd ** 2; v = (theta - mu) ** 2 / sd ** 2
                g_mu = (r_q * u).sum() - n_g * (w * u).sum()
                g_ls = (r_q * v).sum() - n_g * (w * v).sum()
            grad[design.latent] -= g_mu; grad[design.latent + 1] -= g_ls
    return value / n_total, grad / n_total


def start_values(x, obs, design, init=None):
    p = np.clip(x.sum(0) / np.maximum(obs.sum(0), 1), .05, .95)
    base = {"a": np.ones(len(p)), "b": np.clip(-np.log(p / (1 - p)) * 1.2, -3, 3),
            "c": np.full(len(p), .12), "d": np.full(len(p), .97)}
    if init is not None:
        base.update({k: np.asarray(v, float) for k, v in init.items()})
    start = np.zeros(design.size)
    for j, (p_, _, i) in enumerate(design.labels):
        lo, hi = design.bounds_by[p_]
        start[j] = np.clip(base[p_][i], lo + 1e-3, hi - 1e-3)
    return start


def projected_gradient(params, grad, lo, hi, tol=1e-8):
    pg = grad.copy()
    at_lo = params <= lo + tol; at_hi = params >= hi - tol
    pg[at_lo] = np.minimum(grad[at_lo], 0.); pg[at_hi] = np.maximum(grad[at_hi], 0.)
    return pg, at_lo | at_hi


def fit(x, obs, groups, design, quadrature="gauss-hermite", points=41, start=None, max_iter=20000,
        information=True, init=None):
    with _single_thread():
        return _fit(x, obs, groups, design, quadrature, points, start, max_iter, information, init)


def _fit(x, obs, groups, design, quadrature, points, start, max_iter, information, init):
    groups = np.asarray(groups, int)
    quad = Quadrature(quadrature, points)
    parts = _parts(x, obs, groups, design.groups)
    n = len(x)
    bounds = design.bounds()
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    x0 = start_values(x, obs, design, init) if start is None else np.clip(np.asarray(start, float), lo, hi)
    res = minimize(negloglik, x0, args=(design, parts, quad, n), jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": max_iter, "maxfun": 4 * max_iter, "ftol": 1e-15, "gtol": 1e-12, "maxcor": 30})
    params = np.asarray(res.x, float)
    value, score = negloglik(params, design, parts, quad, n)
    pg, on_bound = projected_gradient(params, score, lo, hi, tol=1e-6 * (hi - lo))
    kkt = float(np.abs(pg).max())
    converged = bool(np.isfinite(params).all() and kkt <= SCORE_TOLERANCE)
    # near-boundary = within the shared B1 tolerance of a bound (same rule as the Hessian SE gate)
    at_bound = on_bound | (params - lo <= BOUNDARY_TOLERANCE) | (hi - params <= BOUNDARY_TOLERANCE)
    at_bound[design.latent:] = on_bound[design.latent:]
    regular = bool(converged and not at_bound.any())
    out = {"params": params, "loglik": -value * n, "converged": converged, "regular": regular,
           "at_bound": at_bound, "kkt_per_person": kkt, "iterations": int(res.nit), "design": design,
           "quadrature": quadrature, "points": points, "cov": None, "se_reason": "", "n": n,
           "bound_labels": [design.labels[j] for j in np.where(at_bound[:design.latent])[0]]}
    out["cov_conditional"] = bool(at_bound.any())
    out["linked"] = design.groups == 1 or "anchor" in design.types
    if not out["linked"]:
        out["regular"] = False
        out["se_reason"] = ("Çapa madde yok: grupların ortak ölçeği tanımlanamaz (yalnızca sınırlar nedeniyle sayısal "
                            "olarak tanımlı görünebilir); SH ve çıkarım üretilmez.")
    elif not converged:
        out["se_reason"] = "Kestirim yakınsamadı (KKT ölçütü sağlanmadı)."
    elif information:
        out["cov"], out["se_reason"], out["condition"] = covariance(params, design, parts, quad, n, free=~at_bound)
    if design.groups == 2:
        out["mu_F"], out["sd_F"] = design.latent_of(params, 1)
    return out


def information(params, design, parts, quad, n, free=None):
    """Symmetrized observed information (central-difference Jacobian of the analytic score); columns
    of parameters with free=False are not computed (zero)."""
    m = len(params); info = np.zeros((m, m))
    free = np.ones(m, bool) if free is None else np.asarray(free, bool)
    for j in np.where(free)[0]:
        h = 1e-5 * max(1., abs(params[j]))
        up, dn = params.copy(), params.copy(); up[j] += h; dn[j] -= h
        info[:, j] = (negloglik(up, design, parts, quad, n)[1] - negloglik(dn, design, parts, quad, n)[1]) * n / (2 * h)
    return (info + info.T) / 2


def covariance(params, design, parts, quad, n, max_condition=1e10, free=None):
    """Observed information over the interior parameters; parameters on a bound are held fixed
    (conditional information = sub-matrix of the full information). Rows/columns of fixed
    parameters are NaN in the returned covariance."""
    m = len(params)
    free = np.ones(m, bool) if free is None else np.asarray(free, bool)
    info = information(params, design, parts, quad, n, free)
    sub = info[np.ix_(free, free)]
    eig = np.linalg.eigvalsh(sub)
    cond = float(eig[-1] / eig[0]) if eig[0] > 0 else np.inf
    if not np.isfinite(eig).all() or eig[0] <= 0 or cond > max_condition:
        return None, (f"Gözlenen bilgi matrisi pozitif tanımlı değil veya kötü koşullu (koşul sayısı {cond:.2g}); "
                      "SH üretilemez."), cond
    cov = np.full((m, m), np.nan); cov[np.ix_(free, free)] = np.linalg.inv(sub)
    note = "" if free.all() else (f"{int((~free).sum())} sınır parametresi sabit kabul edilerek koşullu bilgi "
                                  "matrisi kullanıldı.")
    return cov, note, cond


def loglik_at(x, obs, groups, design, params, quadrature="mirt", points=41):
    parts = _parts(x, obs, np.asarray(groups, int), design.groups)
    return -negloglik(np.asarray(params, float), design, parts, Quadrature(quadrature, points), len(x))[0] * len(x)


def fit_single(data, model="3PL", quadrature="gauss-hermite", points=41, bounds=None, information=True, init=None):
    x, obs = as_matrix(data)
    design = Design34(model, ["anchor"] * x.shape[1], 1, bounds)
    return fit(x, obs, np.zeros(len(x), int), design, quadrature, points, information=information, init=init)
