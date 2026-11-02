"""IRT scale linking for dichotomous 1PL/2PL/3PL items.

Clean-room implementation from published formulas (Kolen & Brennan, 2014, ch. 6;
Stocking & Lord, 1983; Haebara, 1980; Loyd & Hoover, 1980; Marco, 1977).
Convention: theta_Y = A*theta_X + B, so form-X items on the Y scale are
a* = a/A, b* = A*b + B, c* = c. Q1 is evaluated on the Y scale with the Y
quadrature, Q2 on the X scale with the X quadrature; Q1Q2 is their sum.
With ``standardized=True`` each part is divided by its weight sum (the
STUIRT/psychometrics default); this changes only the criterion scale.
Status: IMPLEMENTED / PUBLISHED-EXAMPLE MATCHED / NOT CERTIFIED.
"""
import numpy as np
from scipy.optimize import minimize

CRITERIA = ("Q1", "Q2", "Q1Q2")
METHODS = ("stocking_lord", "haebara")


def item_matrix(params):
    """(n, 2|3) array of a, b[, c]; c defaults to 0. Rejects invalid items."""
    p = np.asarray(params, float)
    if p.ndim != 2 or p.shape[1] not in (2, 3) or not len(p) or not np.isfinite(p).all():
        raise ValueError("Madde parametreleri sonlu (n, 2) veya (n, 3) dizi olmalı")
    if p.shape[1] == 2:
        p = np.column_stack([p, np.zeros(len(p))])
    if np.any(p[:, 0] <= 0) or np.any((p[:, 2] < 0) | (p[:, 2] >= 1)):
        raise ValueError("a > 0 ve 0 <= c < 1 olmalı")
    return p


def probability(theta, params, D=1.7):
    p = item_matrix(params)
    t = np.asarray(theta, float)[:, None]
    return p[:, 2] + (1 - p[:, 2]) / (1 + np.exp(-D * p[:, 0] * (t - p[:, 1])))


def to_y_scale(params, A, B):
    p = item_matrix(params).copy()
    p[:, 0], p[:, 1] = p[:, 0] / A, A * p[:, 1] + B
    return p


def to_x_scale(params, A, B):
    p = item_matrix(params).copy()
    p[:, 0], p[:, 1] = p[:, 0] * A, (p[:, 1] - B) / A
    return p


def mean_sigma(x, y):
    """Marco (1977) mean/sigma on common-item difficulties."""
    bx, by = item_matrix(x)[:, 1], item_matrix(y)[:, 1]
    if len(bx) != len(by) or len(bx) < 2 or bx.std() <= 0:
        raise ValueError("En az iki ortak madde ve sıfırdan büyük b yayılımı gerekli")
    A = by.std() / bx.std()
    return dict(A=float(A), B=float(by.mean() - A * bx.mean()), method="mean_sigma")


def mean_mean(x, y):
    """Loyd & Hoover (1980) mean/mean on common-item a and b."""
    px, py = item_matrix(x), item_matrix(y)
    if px.shape != py.shape:
        raise ValueError("Ortak madde sayıları eşit olmalı")
    A = px[:, 0].mean() / py[:, 0].mean()
    return dict(A=float(A), B=float(py[:, 1].mean() - A * px[:, 1].mean()), method="mean_mean")


def criterion(coef, x, y, points_x, weights_x, points_y, weights_y, *, method="stocking_lord",
              part="Q1Q2", D=1.7, standardized=True):
    B, A = coef
    if A <= 0:
        return np.inf
    total = 0.
    for name, theta, w, target, moved in (("Q1", points_y, weights_y, y, to_y_scale(x, A, B)),
                                          ("Q2", points_x, weights_x, x, to_x_scale(y, A, B))):
        if part not in (name, "Q1Q2"):
            continue
        diff = probability(theta, target, D) - probability(theta, moved, D)
        loss = diff.sum(1) ** 2 if method == "stocking_lord" else (diff ** 2).sum(1)
        value = float(np.dot(w, loss))
        total += value / float(np.sum(w)) if standardized else value
    return total


def characteristic_curve_linking(x, y, points_x, weights_x, points_y=None, weights_y=None, *,
                                 method="stocking_lord", part="Q1Q2", D=1.7, standardized=True,
                                 start=(0., 1.)):
    """Stocking–Lord or Haebara linking constants (A, B); never returns unconverged values silently."""
    if method not in METHODS or part not in CRITERIA:
        raise ValueError("Geçersiz yöntem veya ölçüt")
    x, y = item_matrix(x), item_matrix(y)
    if x.shape != y.shape:
        raise ValueError("Ortak madde sayıları eşit olmalı")
    points_y = points_x if points_y is None else points_y
    weights_y = weights_x if weights_y is None else weights_y
    grids = [np.asarray(v, float) for v in (points_x, weights_x, points_y, weights_y)]
    if any(g.ndim != 1 or not np.isfinite(g).all() for g in grids) or len(grids[0]) != len(grids[1]) \
            or len(grids[2]) != len(grids[3]) or min(grids[1].min(), grids[3].min()) < 0:
        raise ValueError("Geçersiz quadrature noktaları/ağırlıkları")
    args = (x, y, *grids)
    kw = dict(method=method, part=part, D=D, standardized=standardized)
    fit = minimize(lambda c: criterion(c, *args, **kw), np.asarray(start, float), method="Nelder-Mead",
                   options=dict(xatol=1e-12, fatol=1e-16, maxiter=20000, maxfev=40000))
    refined = minimize(lambda c: criterion(c, *args, **kw), fit.x, method="BFGS", options=dict(gtol=1e-12))
    best = refined if refined.success and refined.fun <= fit.fun else fit
    if not (fit.success or refined.success) or best.x[1] <= 0:
        raise RuntimeError("Eşitleme ölçütü yakınsamadı")
    return dict(A=float(best.x[1]), B=float(best.x[0]), criterion_value=float(best.fun),
                method=method, part=part, D=D, standardized=standardized,
                status="IMPLEMENTED / PUBLISHED-EXAMPLE MATCHED / NOT CERTIFIED")
