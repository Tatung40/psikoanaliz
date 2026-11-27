import numpy as np, pandas as pd, pytest
from core import mirt_advanced as A

def test_bifactor_pattern_and_pcm_withheld():
    p = A.bifactor_pattern(np.repeat([0, 1, 2], 3))
    assert p.shape == (9, 4) and (p[:, 0] == 1).all() and (p[:, 1:].sum(1) == 1).all()
    with pytest.raises(ValueError): A.bifactor_pattern(np.array([0, 0, 1, 1, 1]))
    with pytest.raises(ValueError, match='WITHHELD'): A.fit_polytomous_bifactor(pd.DataFrame(np.zeros((40, 9), int)), np.repeat([0, 1, 2], 3), 'PCM')

def test_gpcm_bifactor_fit_structure():
    mirt = pytest.importorskip('mirt')
    from mirt.models.polytomous import GeneralizedPartialCredit
    g = np.repeat([0, 1], 3); P = A.bifactor_pattern(g)
    e = GeneralizedPartialCredit(6, n_factors=3, n_categories=3); e.set_parameters(discrimination=P * 1.2, steps=np.tile([-.5, .5], (6, 1)))
    x = pd.DataFrame(mirt.simdata(model=e, n_persons=300, theta=np.random.default_rng(1).standard_normal((300, 3)), seed=4), columns=list('abcdef'))
    o = A.fit_polytomous_bifactor(x, g, 'GPCM', n_quadpts=5, max_iter=60, tol=1e-4)
    a = o['fit'].model.parameters['discrimination']
    assert (a[P == 0] == 0).all() and (a[P == 1] > 0).all() and np.array_equal(o['covariance'], np.eye(3))
    assert o['bifactor_status'] == 'EXPERIMENTAL / NOT CERTIFIED'

def test_high_precision_profile_is_explicit():
    assert A.HIGH_PRECISION == dict(tol=1e-11, max_iter=3000, item_optim_ftol=1e-14, item_optim_maxiter=200)
