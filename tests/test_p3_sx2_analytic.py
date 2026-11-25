"""Independent checks of the Method A S-X2 kernel: Imhof tail, known-parameter eigenvalues, full-functional FD."""
import numpy as np
import pytest
from scipy import stats
from validation.binary_certification.common import truth
from validation.p3_sx2_analytic import kernel as K


def eta_true(model):
    p = truth(model, 12); return np.stack([p[q] for q in "abcd"[:4 if model == "4PL" else 3]], 1).ravel()


def test_imhof_matches_chi_square_and_monte_carlo():
    for df in (3, 5, 9):
        for q in (.5, 3., 12., 25.):
            assert abs(K.imhof_sf(q, np.ones(df)) - stats.chi2.sf(q, df)) < 1e-8
    assert abs(K.imhof_sf(7., np.ones(4), np.full(4, .5)) - stats.ncx2.sf(7., 4, 1.0)) < 1e-8
    lam = np.array([.02, .4, .9, 1.1, 1.3]); z = np.random.default_rng(0).standard_normal((400000, 5))
    q = ((z**2) * lam).sum(1)
    for c in (2., 5., 9.):
        assert abs(K.imhof_sf(c, lam) - (q > c).mean()) < 4 * np.sqrt(.25 / 4e5)


def _map_solve(phat, eta, model, N):
    """Independent Fisher-scoring solve of the penalized MAP equations at pattern distribution phat."""
    for _ in range(50):
        px, score = K.pattern_terms(eta, model); I = (px[:, None] * score).T @ score
        gp, hp = K.log_prior_grad_hess(eta, model); g = phat @ score + gp / N
        step = np.linalg.solve(I - np.diag(hp) / N, g); eta = eta + step
        if np.abs(step).max() < 1e-13: break
    return eta


@pytest.mark.parametrize("model", ["3PL", "4PL"])
def test_influence_function_matches_finite_difference_of_full_functional(model):
    N = 1e9; e0 = eta_true(model); px, _ = K.pattern_terms(e0, model)   # prior/N -> 0: identity point, tests IF algebra
    eta = _map_solve(px, e0, model, N)                        # population MAP functional
    v = np.random.default_rng(1).standard_normal(4096) * px; v -= px * v.sum() / px.sum()
    cond0, _ = K.score_conditionals(eta, model)
    def u(phat):
        e = _map_solve(phat, eta, model, N); cond, _ = K.score_conditionals(e, model)
        return np.array([phat @ (np.isin(K.S, [k]) * (K.X[:, j] - cond[K.S, j])) for j in (0, 6) for k in (3, 7)])
    h = 1e-4; fd = (u(px + h * v) - u(px - h * v)) / (2 * h)
    # analytic: sum_x v_x psi(x), psi = h + G A^-1 s  (re-derived here from the kernel pieces)
    pxe, score = K.pattern_terms(eta, model); I = (pxe[:, None] * score).T @ score
    _, hp = K.log_prior_grad_hess(eta, model); Ai = np.linalg.inv(I - np.diag(hp) / N)
    pi = np.bincount(K.S, weights=pxe, minlength=13); out = []
    for j in (0, 6):
        for k in (3, 7):
            hx = np.isin(K.S, [k]) * (K.X[:, j] - cond0[K.S, j]); d = np.zeros(len(eta))
            for i in range(len(eta)):
                a, b = eta.copy(), eta.copy(); a[i] += 1e-6; b[i] -= 1e-6
                d[i] = -pi[k] * (K.score_conditionals(a, model)[0][k, j] - K.score_conditionals(b, model)[0][k, j]) / 2e-6
            out.append(v @ (hx + score @ Ai.T @ d))
    assert np.allclose(fd, out, rtol=2e-4, atol=1e-9)


def test_known_parameter_unpooled_eigenvalues_are_one():
    model = "3PL"; e = eta_true(model); px, _ = K.pattern_terms(e, model); cond, _ = K.score_conditionals(e, model)
    pi = np.bincount(K.S, weights=px, minlength=13); j = 4
    H = np.stack([np.isin(K.S, [k]) * (K.X[:, j] - cond[K.S, j]) for k in range(1, 12)], 1)
    Sig = (px[:, None] * H).T @ H; w = 1 / (pi[1:12] * cond[1:12, j] * (1 - cond[1:12, j]))
    lam = np.linalg.eigvalsh(np.sqrt(w)[:, None] * Sig * np.sqrt(w)[None])
    assert np.allclose(lam, 1, atol=2e-3)      # 61- vs 41-node quadrature difference only
