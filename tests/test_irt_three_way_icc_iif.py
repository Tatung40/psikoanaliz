"""Three-way deterministic IRT ICC/IIF validation: platform vs scipy vs scalar math.
No personal data, no optimizer dependency. 1PL–4PL including edge theta values.
"""
import math
import numpy as np
import pytest
from scipy.special import expit
from core.irt import probabilities, information

CASES = [
    ("1PL", 1.0, -0.6, 0.0, 1.0),
    ("2PL", 1.7, 0.3, 0.0, 1.0),
    ("3PL", 1.25, 1.1, 0.22, 1.0),
    ("4PL", 2.1, -1.2, 0.17, 0.91),
]
THETAS = np.array([-4., -2., -0.5, 0., 0.7, 2., 4.])

@pytest.mark.parametrize("model,a,b,c,d", CASES)
def test_irt_three_way_icc_and_information(model, a, b, c, d):
    # 1: actual platform functions
    pp = probabilities(THETAS, np.array([a]), np.array([b]),
                       np.array([c]), np.array([d]))[:, 0]
    ii = information(THETAS, np.array([a]), np.array([b]),
                     np.array([c]), np.array([d]))[:, 0]

    # 2: external scipy implementation, no platform helper
    z = expit(a * (THETAS - b))
    scipy_p = c + (d-c) * z
    scipy_dp = (d-c) * a * z * (1-z)
    scipy_i = scipy_dp**2 / (scipy_p*(1-scipy_p))

    # 3: independent scalar math formulation via exp and reciprocal
    scalar_p = []
    scalar_i = []
    for t in THETAS:
        u = math.exp(-a*(float(t)-b))
        s = 1.0/(1.0+u)
        p = c+(d-c)*s
        deriv = a*(p-c)*(d-p)/(d-c)
        scalar_p.append(p)
        scalar_i.append(deriv**2/(p*(1-p)))

    for observed, expected in [
        (pp, scipy_p), (pp, scalar_p),
        (ii, scipy_i), (ii, scalar_i)
    ]:
        np.testing.assert_allclose(observed, expected, rtol=1e-11, atol=1e-12)

    assert np.all((pp > 0) & (pp < 1))
    assert np.all(ii >= 0)
