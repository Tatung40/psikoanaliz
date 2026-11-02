"""IRT true-score and observed-score equating for dichotomous 3PL items.

Clean-room implementation from Kolen & Brennan (2014, ch. 6): true-score
equating with the Kolen linear rule below the sum of guessing parameters, and
observed-score equating with Lord–Wingersky score distributions, synthetic
population weights and equipercentile equating (percentile rank eq. 2.14;
percentile point as the mean of upper/lower points, as in Equating Recipes).
Item parameters must already be on a common scale. Theta values outside the
solvable range are NaN (not ±99 sentinels).
Status: IMPLEMENTED / PUBLISHED-EXAMPLE MATCHED / NOT CERTIFIED.
"""
import numpy as np
from scipy.optimize import brentq
from core.irt_linking import item_matrix, probability


def tcc(theta, params, D=1.7):
    return probability(np.atleast_1d(theta), params, D).sum(1)


def true_score_equating(x, y, D=1.7):
    px, py = item_matrix(x), item_matrix(y)
    cx, cy, nx, ny = px[:, 2].sum(), py[:, 2].sum(), len(px), len(py)
    rows = []
    for s in range(nx + 1):
        if s <= cx:
            theta, eq = np.nan, (s * cy / cx if cx > 0 else 0.)
        elif s >= nx:
            theta, eq = np.nan, float(ny)
        else:
            theta = brentq(lambda t: tcc(t, px, D)[0] - s, -60., 60., xtol=1e-14, rtol=1e-15, maxiter=500)
            eq = float(tcc(theta, py, D)[0])
        rows.append((s, theta, eq))
    s, theta, eq = map(np.asarray, zip(*rows))
    return dict(score=s.astype(int), theta=theta.astype(float), y_equivalent=eq.astype(float),
                status="IMPLEMENTED / PUBLISHED-EXAMPLE MATCHED / NOT CERTIFIED")


def lord_wingersky(params, theta, weights, D=1.7):
    """Marginal number-correct distribution over a user-supplied latent quadrature."""
    w = np.asarray(weights, float)
    if w.ndim != 1 or len(w) != len(np.atleast_1d(theta)) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Geçersiz latent dağılım ağırlıkları")
    P = probability(np.atleast_1d(theta), params, D)
    dist = np.ones((len(w), 1))
    for j in range(P.shape[1]):
        p = P[:, [j]]
        dist = np.hstack([dist * (1 - p), np.zeros((len(w), 1))]) + np.hstack([np.zeros((len(w), 1)), dist * p])
    return (w / w.sum()) @ dist


def percentile_rank(cdf, x):
    """Kolen & Brennan eq. 2.14 for integer scores 0..K (increment 1)."""
    below = cdf[x - 1] if x > 0 else 0.
    return 100 * (below + .5 * (cdf[x] - below))


def percentile_point(cdf, pr):
    p, n = pr / 100, len(cdf)
    if p <= 1e-8:
        upper = next((i for i in range(n) if cdf[i] > 1e-8), n) - .5
        return (upper - .5) / 2
    if p >= 1 - 1e-8:
        j = next((j for j in range(n - 1, -1, -1) if cdf[j] < 1 - 1e-8), -1)
        return ((n - .5) + (j + 1.5)) / 2
    if cdf[0] > p:
        return p / cdf[0] - .5
    i = next(i for i in range(1, n) if cdf[i] > p)
    upper = (p - cdf[i - 1]) / (cdf[i] - cdf[i - 1]) + (i - .5) if cdf[i] != cdf[i - 1] else i - .5
    j = next((j for j in range(n - 2, -1, -1) if cdf[j] < p), -1)
    lower = (p - cdf[j]) / (cdf[j + 1] - cdf[j]) + (j + .5) if j >= 0 and cdf[j + 1] != cdf[j] else j + .5
    return (upper + lower) / 2


def observed_score_equating(x, quad_x, y, quad_y, *, w1=1., D=1.7):
    """quad_* = (theta, weights) latent distribution of population 1 (X) and 2 (Y)."""
    if not 0 <= w1 <= 1:
        raise ValueError("w1 [0, 1] aralığında olmalı")
    (tx, wx), (ty, wy) = quad_x, quad_y
    f = w1 * lord_wingersky(x, tx, wx, D) + (1 - w1) * lord_wingersky(x, ty, wy, D)
    g = w1 * lord_wingersky(y, tx, wx, D) + (1 - w1) * lord_wingersky(y, ty, wy, D)
    cf, cg = np.cumsum(f), np.cumsum(g)
    eq = np.array([percentile_point(cg, percentile_rank(cf, s)) for s in range(len(f))])
    return dict(score=np.arange(len(f)), x_distribution=f, y_distribution=g, y_equivalent=eq,
                status="IMPLEMENTED / PUBLISHED-EXAMPLE MATCHED / NOT CERTIFIED")
