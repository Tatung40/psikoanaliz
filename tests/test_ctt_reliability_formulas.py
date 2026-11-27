"""Independent formula checks: one-factor ML omega (exact congeneric covariance), split-half, Guttman, Feldt."""
import numpy as np
import pandas as pd
from core.reliability import compute_reliability_indices
from core import reliability_extended as RX


def exact_data(R, n=400, seed=0):
    """Data whose sample correlation matrix equals R exactly."""
    z = np.random.default_rng(seed).standard_normal((n, len(R))); z -= z.mean(0)
    z = z @ np.linalg.inv(np.linalg.cholesky(np.cov(z, rowvar=False))).T
    return pd.DataFrame(z @ np.linalg.cholesky(R).T + 3, columns=[f"m{i}" for i in range(len(R))])


def test_omega_is_one_factor_ml_not_principal_component():
    lam = np.array([.4, .5, .6, .7, .8, .55]); R = np.outer(lam, lam); np.fill_diagonal(R, 1)
    rel = compute_reliability_indices(exact_data(R), [f"m{i}" for i in range(6)], is_dichotomous=False)
    truth = lam.sum() ** 2 / (lam.sum() ** 2 + (1 - lam ** 2).sum())
    assert abs(rel["mcdonald_omega"] - truth) <= 1e-4
    ev, vec = np.linalg.eigh(R); pc = np.abs(vec[:, -1]) * np.sqrt(ev[-1])
    pca = pc.sum() ** 2 / (pc.sum() ** 2 + np.maximum(0, 1 - pc ** 2).sum())
    assert pca - truth > .01                               # the replaced PCA approximation is biased upward


def test_omega_heywood_is_not_reported():
    lam = np.array([.999, .3, .3, .3]); R = np.outer(lam, lam); np.fill_diagonal(R, 1)
    rel = compute_reliability_indices(exact_data(R), [f"m{i}" for i in range(4)], is_dichotomous=False)
    assert rel["mcdonald_omega"] is None or abs(rel["mcdonald_omega"] - (lam.sum()**2/(lam.sum()**2+(1-lam**2).sum()))) <= 1e-3


def test_split_half_and_guttman_lambda4():
    rng = np.random.default_rng(1); th = rng.normal(size=500)
    x = pd.DataFrame(np.clip(np.round(th[:, None] + rng.normal(0, 1, (500, 8)) + 2), 0, 4), columns=[f"m{i}" for i in range(8)])
    rel = compute_reliability_indices(x, list(x), is_dichotomous=False)
    a, b = x.iloc[:, 0::2].sum(1), x.iloc[:, 1::2].sum(1); r = np.corrcoef(a, b)[0, 1]
    assert abs(rel["spearman_brown_odd_even"] - 2 * r / (1 + r)) < 1e-4
    assert abs(rel["guttman_lambda4"] - 2 * (1 - (a.var() + b.var()) / (a + b).var())) < 1e-4


def test_guttman_and_feldt_family_against_loops():
    rng = np.random.default_rng(2); x = rng.normal(size=(300, 5)) + rng.normal(size=(300, 1)); S = np.cov(x, rowvar=False); k = 5; V = S.sum()
    out = RX.reliability_family(S)
    l1 = 1 - np.trace(S) / V; c2 = sum(S[i, j] ** 2 for i in range(k) for j in range(k) if i != j)
    assert abs(out["lambda2"] - (l1 + np.sqrt(k / (k - 1) * c2) / V)) < 1e-12 and abs(out["lambda3"] - k / (k - 1) * l1) < 1e-12
    e2 = [np.var(x[:, j] - np.c_[np.ones(300), np.delete(x, j, 1)] @ np.linalg.lstsq(np.c_[np.ones(300), np.delete(x, j, 1)], x[:, j], rcond=None)[0], ddof=1) for j in range(k)]
    assert abs(out["lambda6"] - (1 - sum(e2) / V)) < 1e-10
    sx = S.sum(1); assert abs(out["feldt_brennan"] - V * (V - np.trace(S)) / (V ** 2 - (sx ** 2).sum())) < 1e-12


def test_raw_omega_separate_estimand_exact_congeneric():
    lam = np.array([.4, .5, .6, .7, .8, .55]); sd = np.array([1., 2., .5, 1.5, 1., 3.])
    R = np.outer(lam, lam); np.fill_diagonal(R, 1)
    x = exact_data(R) * sd
    rel = compute_reliability_indices(x, list(x), is_dichotomous=False)
    lr, pr = lam * sd, (1 - lam ** 2) * sd ** 2
    assert abs(rel["mcdonald_omega_raw"] - lr.sum() ** 2 / (lr.sum() ** 2 + pr.sum())) <= 1e-4
    assert abs(rel["mcdonald_omega"] - lam.sum() ** 2 / (lam.sum() ** 2 + (1 - lam ** 2).sum())) <= 1e-4
