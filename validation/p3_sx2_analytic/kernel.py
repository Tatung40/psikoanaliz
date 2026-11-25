"""Exact-pattern (J=12) nuisance-adjusted S-X2 residual covariance and generalized chi-square tail.

Method A (S-X2 itself, corrected reference distribution):
  u_jg(p_hat, eta) = sum_x p_hat_x 1[s(x) in g] (x_j - P_j(1 | s(x); eta))        (linear in p_hat)
  S-X2_j = N sum_g w_jg u_jg^2,   w_jg = n_g / (e1_jg e0_jg)                          (identical algebra)
  IF(x) = h(x) + G A^{-1} score(x),  G = d E[u]/d eta,  A = I_exp + (-H_prior)/N      (MAP M-estimator)
  sqrt(N) u -> N(delta, Sigma),  delta = sqrt(N) G A^{-1} grad_prior / N  (prior-induced residual bias)
  S-X2 ~ sum_i lambda_i (z_i + nu_i)^2, lambda = eig(W^1/2 Sigma W^1/2)  -> Imhof tail.
Grouping: production row pooling as realised in the data (conditioning; validated empirically).
"""
import numpy as np
from scipy import integrate
from scipy.special import expit
from mirt.scoring._common import build_quadrature
from core.irt_binary_patterns import PATTERNS

J = 12
X = PATTERNS.astype(float)                     # 4096 x 12
S = X.sum(1).astype(int)
BITS = 1 << np.arange(J)
INDEX = np.empty(4096, int); INDEX[PATTERNS.astype(int) @ BITS] = np.arange(4096)


def pattern_counts(x):
    return np.bincount(INDEX[np.asarray(x, int) @ BITS], minlength=4096).astype(float)


def quad(n):
    t, w = build_quadrature(n_quadpts=n, n_factors=1, prior_mean=None, prior_cov=None)
    return np.asarray(t, float).ravel(), np.asarray(w, float).ravel()


T61, W61 = quad(61)
T41, W41 = quad(41)


def unpack(eta, model):
    k = 4 if model == "4PL" else 3
    e = np.asarray(eta, float).reshape(J, k)
    return e[:, 0], e[:, 1], e[:, 2], (e[:, 3] if k == 4 else np.ones(J))


def irf(eta, model, t):
    a, b, c, d = unpack(eta, model)
    z = expit(a * (t[:, None] - b)); return c + (d - c) * z, z


def irf_grad(eta, model, t):
    """dP/d(a,b,c,d) at nodes: arrays nodes x J x k."""
    a, b, c, d = unpack(eta, model); z = expit(a * (t[:, None] - b)); dz = z * (1 - z)
    g = [(d - c) * dz * (t[:, None] - b), -(d - c) * dz * a, 1 - z]
    if model == "4PL": g.append(z)
    return np.stack(g, 2)


def pattern_terms(eta, model):
    """Marginal pattern probabilities (61-node estimator quadrature) and per-pattern likelihood scores."""
    P, _ = irf(eta, model, T61); P = np.clip(P, 1e-300, 1 - 1e-16)
    ll = X @ np.log(P).T + (1 - X) @ np.log1p(-P).T               # 4096 x Q
    m = ll.max(1, keepdims=True); L = np.exp(ll - m) * W61
    px = L.sum(1); post = L / px[:, None]; px = px * np.exp(m[:, 0])
    dP = irf_grad(eta, model, T61)                                 # Q x J x k
    resid = (X[:, None, :] - P[None]) / (P * (1 - P))[None]        # 4096 x Q x J
    score = np.einsum("xq,xqj,qjk->xjk", post, resid, dP).reshape(len(X), -1)
    return px, score


def log_prior_grad_hess(eta, model):
    a, b, c, d = unpack(eta, model)
    ga = -1 / a - np.log(a) / (.25 * a); ha = 1 / a**2 - (1 - np.log(a)) / (.25 * a**2)
    gb = -b / 4; hb = -np.ones(J) / 4
    gc = 1 / c - 7 / (1 - c); hc = -1 / c**2 - 7 / (1 - c)**2
    g = [ga, gb, gc]; h = [ha, hb, hc]
    if model == "4PL":
        g.append(7 / d - 1 / (1 - d)); h.append(-7 / d**2 - 1 / (1 - d)**2)
    return np.stack(g, 1).ravel(), np.stack(h, 1).ravel()


def score_conditionals(eta, model):
    """P_j(1 | total score k) for k=0..J (41-node S-X2 quadrature) and score distribution."""
    P, _ = irf(eta, model, T41)
    Q = len(T41); dist = np.ones((Q, 1)); pre = []
    for j in range(J):
        pre.append(dist); new = np.zeros((Q, dist.shape[1] + 1)); new[:, :-1] += dist * (1 - P[:, j:j + 1]); new[:, 1:] += dist * P[:, j:j + 1]; dist = new
    marg = W41 @ dist
    cond = np.zeros((J + 1, J))
    for j in range(J):
        rest = np.ones((Q, 1))
        for i in range(J):
            if i == j: continue
            new = np.zeros((Q, rest.shape[1] + 1)); new[:, :-1] += rest * (1 - P[:, i:i + 1]); new[:, 1:] += rest * P[:, i:i + 1]; rest = new
        joint = W41 @ (rest * P[:, j:j + 1])                         # P(item j=1, rest=k-1)
        cond[1:, j] = joint / marg[1:]
    return cond, marg


def pooling(observed, expected, minimum=1.0):
    """Replicates production _pool_score_rows grouping; returns list of score-index groups (scores 1..J-1)."""
    minimum *= 1.0 - 1e-9
    groups = [[k] for k in range(1, J)]
    obs, exp_ = observed[1:-1].copy(), expected[1:-1].copy()
    keep = obs.sum(1) > 0
    groups = [g for g, z in zip(groups, keep) if z]; obs, exp_ = obs[keep], exp_[keep]
    while len(exp_) > 1:
        sp = np.flatnonzero(exp_.min(1) < minimum)
        if sp.size == 0: break
        r = int(sp[0]); nb = 1 if r == 0 else r - 1 if r == len(exp_) - 1 else (r - 1 if obs[r - 1].sum() <= obs[r + 1].sum() else r + 1)
        obs[nb] += obs[r]; exp_[nb] += exp_[r]; groups[nb] = sorted(groups[nb] + groups[r])
        obs = np.delete(obs, r, 0); exp_ = np.delete(exp_, r, 0); del groups[r]
    sparse = bool((exp_ < minimum).any())
    return groups, sparse


def imhof_sf(q, lam, nc=None):
    """P(sum lam_i (z_i+nc_i)^2 > q): Imhof (1961) inversion, log-stable integrand, geometric panels up to
    Imhof's truncation bound (tail < 1e-10). Validated for >=3 non-negligible weights (production spectra)."""
    lam = np.asarray(lam, float); d2 = np.zeros_like(lam) if nc is None else np.asarray(nc, float)**2
    if len(lam) < 3: raise ValueError("tail algorithm validated only for >=3 positive weights")
    def f(u):
        lu = lam * u; th = .5 * np.sum(np.arctan(lu) + d2 * lu / (1 + lu**2)) - .5 * q * u
        lr = .25 * np.sum(np.log1p(lu**2)) + .5 * np.sum(d2 * lu**2 / (1 + lu**2))
        return np.sin(th) / u * np.exp(-lr)
    k = len(lam); U = (2 / (np.pi * k * 1e-10 * np.prod(np.sqrt(lam))))**(2 / k)
    per = 2 * np.pi / max(q, 1e-3); edges = np.concatenate([[0], np.geomspace(per / 50, max(U, per), 400)])
    v = sum(integrate.quad(f, a, b, limit=200, epsabs=1e-13)[0] for a, b in zip(edges[:-1], edges[1:]))
    return float(min(max(.5 + v / np.pi, 0.0), 1.0))


def _safe(q, lam, nc=None):
    try: return imhof_sf(q, lam, nc) if np.isfinite(q) else float('nan')
    except Exception: return float('nan')


def analyse(x, eta, model, include_bias=True, eps=1e-6):
    """Production S-X2 (recomputed), Method A corrected p-values, per item. x: N x J responses."""
    x = np.asarray(x, int); N = len(x)
    phat = pattern_counts(x) / N
    px, score = pattern_terms(eta, model)
    I = (px[:, None] * score).T @ score
    gp, hp = log_prior_grad_hess(eta, model)
    A = I - np.diag(hp) / N
    Ainv = np.linalg.inv(A)
    cond, marg = score_conditionals(eta, model)
    pi = np.bincount(S, weights=px, minlength=J + 1)               # model score distribution (61-node marginal)
    # derivative of conditionals wrt eta (central differences of the exact recursion)
    dcond = np.zeros((J + 1, J, len(eta)))
    for i in range(len(eta)):
        e1 = np.array(eta, float); e2 = e1.copy(); e1[i] += eps; e2[i] -= eps
        dcond[:, :, i] = (score_conditionals(e1, model)[0] - score_conditionals(e2, model)[0]) / (2 * eps)
    nk = np.bincount(x.sum(1), minlength=J + 1).astype(float)
    out = []
    for j in range(J):
        o1 = np.bincount(x.sum(1), weights=x[:, j], minlength=J + 1)
        O = np.c_[nk - o1, o1]; E = np.c_[nk * (1 - cond[:, j]), nk * cond[:, j]]
        groups, sparse = pooling(O, E)
        G = len(groups); npar = len(eta) // J
        H = np.zeros((4096, G)); Gm = np.zeros((G, len(eta))); w = np.zeros(G); u = np.zeros(G); sx = 0.0
        for g, ks in enumerate(groups):
            m = np.isin(S, ks)
            H[:, g] = m * (X[:, j] - cond[S, j])
            Gm[g] = -sum(pi[k] * dcond[k, j] for k in ks)
            o, e = O[ks].sum(0), E[ks].sum(0); sx += ((o - e)**2 / e).sum()
            ng = nk[ks].sum() / N; e1 = e[1] / N; e0 = e[0] / N
            w[g] = ng / (e1 * e0); u[g] = phat @ H[:, g]
        psi = H + score @ Ainv.T @ Gm.T                                # 4096 x G influence
        mu = px @ psi; Sig = (px[:, None] * psi).T @ psi - np.outer(mu, mu)
        Wh = np.sqrt(w); M = Wh[:, None] * Sig * Wh[None]
        lam, V = np.linalg.eigh(M); lam = np.clip(lam, 0, None)
        delta = np.sqrt(N) * (Gm @ Ainv @ gp) / N if include_bias else np.zeros(G)
        nc = np.zeros(G); pos = lam > 1e-12 * lam.max()
        nc[pos] = (V.T @ (Wh * delta))[pos] / np.sqrt(lam[pos])
        df = max(G - 1 * 0 - npar, 0)  # production df: G contrasts - npar
        out.append(dict(sx=float(sx), sx_from_u=float(N * (w * u**2).sum()), groups=groups, sparse=sparse, df=df,
                        lam=lam.tolist(), p_A=_safe(sx, lam[pos], nc[pos]), p_A_nobias=_safe(sx, lam[pos]),
                        mean_E=float(lam.sum() + (nc**2 * lam).sum())))
    return out
