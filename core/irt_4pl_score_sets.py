"""4PL c/d frequentist confidence sets by Neyman C(alpha) efficient-score test inversion (closure K).

For psi = c_j or d_j: nuisance path eta(psi) = MAP + H^-1 e_i (psi - MAP_i)/(H^-1)_ii (H = exact expected information at
the MAP, 4096 patterns, 61 nodes), clipped to the native box. T(psi) = N g^2 / v with efficient score
g = s_i - I_il I_ll^-1 s_l (unpenalised mean marginal score at eta(psi), empirical pattern distribution) and
v = I_ii - I_il I_ll^-1 I_li. Set = union of grid cells (G=26 over the box, half-width h/2) whose centre has T <= 3.8415.
May be empty, disconnected or the whole box; whole-box sets are reported as uninformative. MAP estimates unchanged.
"""
import numpy as np
from scipy.special import expit

J = 12; G = 26; CRIT = 3.841458820694124; BOXQ = {2: (0.0, .5), 3: (.5, 1.0)}
LO = np.array([.05, -6, 1e-6, .5 + 1e-6] * J); HI = np.array([6, 6, .5 - 1e-6, 1 - 1e-6] * J)
_X = ((np.arange(2 ** J)[:, None] >> np.arange(J)) & 1).astype(float); _BITS = 1 << np.arange(J)


def _quad():
    from mirt.scoring._common import build_quadrature
    t, w = build_quadrature(n_quadpts=61, n_factors=1, prior_mean=None, prior_cov=None)
    return np.asarray(t, float).ravel(), np.asarray(w, float).ravel()


def terms(eta, t, w):
    e = np.asarray(eta, float).reshape(J, 4); a, b, c, d = e.T
    z = expit(a * (t[:, None] - b)); P = np.clip(c + (d - c) * z, 1e-300, 1 - 1e-16); dz = z * (1 - z)
    dP = np.stack([(d - c) * dz * (t[:, None] - b), -(d - c) * dz * a, 1 - z, z], 2)
    ll = _X @ np.log(P).T + (1 - _X) @ np.log1p(-P).T; m = ll.max(1, keepdims=True); L = np.exp(ll - m) * w
    px = L.sum(1); post = L / px[:, None]; px = px * np.exp(m[:, 0])
    sc = np.einsum("xq,xqj,qjk->xjk", post, (_X[:, None, :] - P[None]) / (P * (1 - P))[None], dP).reshape(len(_X), -1)
    return px, sc


def statistic(eta, phat, i, N, t, w):
    px, sc = terms(eta, t, w); s = phat @ sc; I = (px[:, None] * sc).T @ sc
    lam = np.r_[0:i, i + 1:len(eta)]; b = np.linalg.solve(I[np.ix_(lam, lam)], I[lam, i])
    g = s[i] - b @ s[lam]; v = I[i, i] - b @ I[lam, i]
    return float(N * g * g / v)


def confidence_sets(eta_map, responses, items=range(J), params=(2, 3), truth=None):
    """Per (item, param): accepted grid, union-of-cells intervals, whole/empty flags; truth coverage if given."""
    x = np.asarray(responses, int); N = len(x); t, w = _quad()
    phat = np.bincount(x @ _BITS, minlength=2 ** J) / N
    eta = np.asarray(eta_map, float).ravel(); px, sc = terms(eta, t, w); I = (px[:, None] * sc).T @ sc
    ev = np.linalg.eigvalsh(I)
    if not np.all(np.isfinite(I)) or ev[0] <= 0 or ev[-1] / ev[0] > 1e12:
        raise np.linalg.LinAlgError("expected information not positive definite / cond > 1e12")
    Hi = np.linalg.inv(I); anchor = np.clip(eta, LO, HI); out = []
    for j in items:
        for q in params:
            i = 4 * j + q; lo, hi = BOXQ[q]; grid = np.linspace(lo, hi, G); h = (hi - lo) / (G - 1)
            def T(psi):
                e = np.clip(anchor + Hi[:, i] * (psi - anchor[i]) / Hi[i, i], LO, HI); e[i] = np.clip(psi, LO[i], HI[i])
                return statistic(e, phat, i, N, t, w)
            acc = np.array([T(p) <= CRIT for p in grid]); runs = []
            for k, a in enumerate(acc):
                if a and (k == 0 or not acc[k - 1]): start = max(grid[k] - h / 2, lo)
                if a and (k == G - 1 or not acc[k + 1]): runs.append((float(start), float(min(grid[k] + h / 2, hi))))
            rec = dict(item=int(j), param="abcd"[q], intervals=runs, whole=bool(acc.all()), empty=not acc.any(),
                       width=float(sum(b - a for a, b in runs)))
            if truth is not None:
                tv = float(np.asarray(truth).ravel()[i]); rec["covered_set"] = any(a <= tv <= b for a, b in runs)
                rec["covered_exact"] = bool(T(tv) <= CRIT)
            out.append(rec)
    return out
