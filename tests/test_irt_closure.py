"""No-fit numerical contracts before scientific execution."""
import numpy as np
import pytest
from scipy.stats import norm
from validation.irt_closure.bootstrap import binding_inversion
from validation.irt_closure.patterns import PATTERNS,SCORES,moments,score_mass,oracle,interval_metrics,exact_score_intervals
from validation.binary_certification.common import truth


def test_full_joint_gaussian_binding_inversion():
    I=np.array([[3.,.5],[.5,2.]])
    penalty=np.diag([4.,5.]);V=np.linalg.inv(I+penalty)
    e=np.array([.2,-.1]);prior=-penalty@e
    scores=np.array([[1.,2.],[-1.,0.],[2.,-1.],[-2.,-1.]])
    stars=e+(scores+prior)@V.T
    draws,J=binding_inversion(e,stars,scores,V,I,prior)
    np.testing.assert_allclose(J,V@I,atol=1e-13)
    expected=e+np.linalg.solve(V@I,(e-stars).T).T
    np.testing.assert_allclose(draws,expected,atol=1e-13)
    assert abs(J[0,1])>0


@pytest.mark.parametrize("model",["3PL","4PL"])
def test_pattern_probability_and_posterior_moment_quadrature(model):
    p=truth(model,12)
    m,a,s=moments(p,241);m2,a2,s2=moments(p,481)
    assert abs(m.sum()-1)<1e-10
    assert abs(m-m2).sum()<1e-8
    np.testing.assert_allclose(a,a2,atol=1e-6,rtol=0)
    np.testing.assert_allclose(s,s2,atol=1e-6,rtol=0)
    assert abs(m@a)<1e-9


def test_score_polynomial_matches_pattern_enumeration():
    from scipy.special import expit
    p=expit(np.linspace(-1,1,12))
    mass=np.exp(PATTERNS@np.log(p)+(1-PATTERNS)@np.log1p(-p))
    np.testing.assert_allclose(np.bincount(SCORES,weights=mass,minlength=13),score_mass(p),atol=1e-14)


@pytest.mark.parametrize("model",["3PL","4PL"])
def test_exact_known_item_score_inversion_coverage(model):
    reference=oracle(model)
    ci=exact_score_intervals(reference[0])[SCORES]
    result=interval_metrics(ci,reference)
    assert min(result["conditional_theta"])>=.95-1e-9
    assert result["marginal"]>=.95-1e-6
    assert min(result["tails"])>=.95-1e-6
    credible=interval_metrics(reference[5],reference)
    assert credible["marginal"]==pytest.approx(.95,abs=1e-6)


def test_gaussian_inverse_does_not_relabel_map_sampling_se():
    alpha=.2
    sd_MAP=alpha
    J=alpha
    assert sd_MAP/J==pytest.approx(1.)
    assert sd_MAP!=sd_MAP/J
