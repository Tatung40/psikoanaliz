"""Gradient-projection oblique rotation (quartimin, oblimin, geomin).

Clean-room implementation of the GPF oblique algorithm of Bernaards & Jennrich
(2005, Educational and Psychological Measurement 65, 676-696) with the
published criterion gradients (Jennrich, 2002; Yates, 1987 geomin, delta=.01).
No Kaiser normalization, no reflection/reordering: loadings L = A (T')^-1 and
factor correlations Phi = T'T, as in GPArotation::GPFoblq.
Status: IMPLEMENTED / PUBLISHED-EXAMPLE MATCHED / NOT CERTIFIED.
"""
import numpy as np


def _oblimin(L, gamma=0.):
    p, k = L.shape
    L2 = L ** 2
    N = np.ones((k, k)) - np.eye(k)
    X = (np.eye(p) - gamma / p * np.ones((p, p))) @ L2 @ N
    return float((L2 * X).sum() / 4), L * X


def _geomin(L, delta=.01):
    k = L.shape[1]
    L2 = L ** 2 + delta
    pro = np.exp(np.log(L2).sum(1) / k)
    return float(pro.sum()), (2 / k) * (L / L2) * pro[:, None]


def criterion(name, gamma=0., delta=.01):
    if name in ("quartimin", "oblimin"):
        g = 0. if name == "quartimin" else gamma
        return lambda L: _oblimin(L, g)
    if name == "geomin":
        return lambda L: _geomin(L, delta)
    raise ValueError("Desteklenen ölçütler: quartimin, oblimin, geomin")


def gpa_oblique(A, method="quartimin", *, gamma=0., delta=.01, max_iter=1000, tol=1e-5):
    A = np.asarray(A, float)
    if A.ndim != 2 or A.shape[1] < 2 or A.shape[0] <= A.shape[1] or not np.isfinite(A).all():
        raise ValueError("Sonlu p x k (p > k >= 2) yük matrisi gerekli")
    vgq = criterion(method, gamma, delta)
    T = np.eye(A.shape[1])
    L = A @ np.linalg.inv(T).T
    f, Gq = vgq(L)
    G = -(L.T @ Gq @ np.linalg.inv(T)).T
    step, converged, s = 1., False, np.inf
    for iteration in range(max_iter + 1):
        Gp = G - T @ np.diag((T * G).sum(0))
        s = float(np.sqrt((Gp ** 2).sum()))
        if s < tol:
            converged = True
            break
        step *= 2
        for _ in range(11):
            X = T - step * Gp
            Tt = X / np.sqrt((X ** 2).sum(0))
            L = A @ np.linalg.inv(Tt).T
            ft, Gq = vgq(L)
            if f - ft > .5 * s ** 2 * step:
                break
            step /= 2
        T, f = Tt, ft
        G = -(L.T @ Gq @ np.linalg.inv(Tt)).T
    L = A @ np.linalg.inv(T).T
    return dict(loadings=L, phi=T.T @ T, rotation=T, criterion=f, iterations=iteration,
                converged=converged, gradient_norm=s, method=method,
                status="IMPLEMENTED / PUBLISHED-EXAMPLE MATCHED / NOT CERTIFIED")
