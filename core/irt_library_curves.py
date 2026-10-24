"""Pure, explicit 1PL–4PL curve calculations from named parameters.

Only accepts parameters explicitly mapped to the IRT logistic metric.
Never guesses whether library outputs use slope/intercept or a/b.
"""
import numpy as np
from scipy.special import expit


def binary_curves(theta, *, a, b, c=None, d=None):
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 1 or not np.isfinite(theta).all():
        raise ValueError("theta must be a finite one-dimensional grid")
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("a and b must have matching nonempty lengths")
    c = np.zeros_like(a) if c is None else np.asarray(c, dtype=float).reshape(-1)
    d = np.ones_like(a) if d is None else np.asarray(d, dtype=float).reshape(-1)
    if any(len(v) != len(a) for v in (c, d)):
        raise ValueError("c and d must match a/b lengths")
    if not all(np.isfinite(v).all() for v in (a, b, c, d)):
        raise ValueError("parameters must be finite")
    if np.any(a <= 0) or np.any(c < 0) or np.any(d > 1) or np.any(c >= d):
        raise ValueError("invalid 4PL parameter constraints")
    z = a[None, :] * (theta[:, None] - b[None, :])
    s = expit(z)
    p = c[None, :] + (d-c)[None, :] * s
    derivative = (d-c)[None, :] * a[None, :] * s * (1-s)
    information = np.divide(derivative**2, p*(1-p),
                            out=np.zeros_like(p), where=(p > 0) & (p < 1))
    return {"theta": theta, "probability": p,
            "information": information, "test_information": information.sum(axis=1)}
