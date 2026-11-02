"""Covariance-based reliability coefficients missing from core.reliability.

Clean-room from published formulas: Guttman (1945) lambda1, lambda2,
lambda3 (= alpha), lambda5, lambda6 (covariance form with residual variances
1/(S^-1)_jj); Feldt–Gilmer (Gilmer & Feldt, 1983) and Feldt–Brennan classical
congeneric (Feldt & Brennan, 1989). Complete data only; no imputation.
Status: IMPLEMENTED / PUBLISHED-EXAMPLE MATCHED (lambdas) / ANALYTIC IDENTITY
CHECKED (Feldt–Gilmer, Feldt–Brennan) / NOT CERTIFIED.
"""
import numpy as np


def _matrix(cov):
    S = np.asarray(cov, float)
    if S.ndim != 2 or S.shape[0] != S.shape[1] or S.shape[0] < 2 or not np.isfinite(S).all():
        raise ValueError("Sonlu kare kovaryans matrisi gerekli (k >= 2)")
    if not np.allclose(S, S.T, rtol=1e-10, atol=1e-12) or np.any(np.diag(S) <= 0):
        raise ValueError("Kovaryans simetrik ve pozitif varyanslı olmalı")
    return (S + S.T) / 2


def covariance_from_data(data):
    x = np.asarray(data, float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Eksiksiz sayısal veri gerekli; eksik veri için önce açık bir karar verin")
    return np.cov(x, rowvar=False, ddof=1)


def guttman_lambdas(cov):
    S = _matrix(cov)
    k, var = len(S), S.sum()
    off = S - np.diag(np.diag(S))
    l1 = 1 - np.trace(S) / var
    l2 = l1 + np.sqrt(k / (k - 1) * (off ** 2).sum()) / var
    l3 = k / (k - 1) * l1
    l5 = l1 + 2 * np.sqrt((off ** 2).sum(0).max()) / var
    eig = np.linalg.eigvalsh(S)
    l6 = 1 - (1 / np.diag(np.linalg.inv(S))).sum() / var if eig[0] > 1e-12 * eig[-1] else np.nan
    return dict(lambda1=float(l1), lambda2=float(l2), lambda3=float(l3), lambda5=float(l5), lambda6=float(l6))


def feldt_brennan(cov):
    S = _matrix(cov)
    var = S.sum()
    return float(var * (var - np.trace(S)) / (var ** 2 - (S.sum(1) ** 2).sum()))


def feldt_gilmer(cov):
    S = _matrix(cov)
    k = len(S)
    if k < 3:
        return np.nan
    rows = S.sum(1)
    ell = int(np.argmax(rows - np.diag(S)))
    d = np.ones(k)
    for i in range(k):
        if i != ell:
            d[i] = (rows[i] - S[i, ell] - S[i, i]) / (rows[ell] - S[i, ell] - S[ell, ell])
    var = S.sum()
    return float(d.sum() ** 2 / (d.sum() ** 2 - (d ** 2).sum()) * (var - np.trace(S)) / var)


def reliability_family(cov):
    out = guttman_lambdas(cov)
    out.update(feldt_gilmer=feldt_gilmer(cov), feldt_brennan=feldt_brennan(cov),
               status="IMPLEMENTED / NOT CERTIFIED")
    return out
