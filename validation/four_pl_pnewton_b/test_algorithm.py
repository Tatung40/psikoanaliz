import numpy as np
from validation.four_pl_pnewton_b.algorithm import maximize, projected_score


def test_bounded_concave_quadratic_reaches_kkt_point():
    Q = np.array([[4., 1., 0.], [1., 3., .5], [0., .5, 2.]]); c = np.array([10., -6., 1.])  # unconstrained max outside box
    lo, hi = np.zeros(3), np.ones(3)
    f = lambda x: c @ x - .5 * x @ Q @ x; g = lambda x: c - Q @ x; H = lambda x: Q
    x, fx, h = maximize(f, g, H, np.full(3, .5), lo, hi, tol=1e-10)
    assert h[-1]["stop"] == "STATIONARY" and np.all((x >= lo) & (x <= hi))
    assert np.allclose(x, [1., 0., .5]) and abs(projected_score(x, g(x), lo, hi)).max() <= 1e-10
    assert all(b["f"] >= a["f"] for a, b in zip(h, h[1:]))          # nondecreasing objective


def test_gradient_direction_is_ascent_for_negative_hessian_convention():
    Q = np.diag([2., 5.]); f = lambda x: -.5 * x @ Q @ x; g = lambda x: -Q @ x
    x, _, h = maximize(f, g, lambda x: Q, np.array([.3, -.2]), np.full(2, -1.), np.full(2, 1.), tol=1e-12)
    assert h[0]["ascent_gFdF"] > 0 and np.allclose(x, 0)
