"""Analytic coupling obstruction for the exact registered 4PL alternative."""
import math
import numpy as np
from scipy import integrate,stats
from scipy.special import expit


def test_registered_4pl_alternative_has_an_indistinguishable_feasible_null():
    m=integrate.quad(lambda t:(.1+.8*expit(1.6*(t+1.25)))*stats.norm.pdf(t),-12,12,epsabs=1e-13)[0]
    assert .5<m<.9  # therefore the constructed upper asymptote is native-box feasible
    t=np.linspace(-30,30,1001)
    assert np.all(expit(-5*(t+6))<=np.exp(-5*(t+6))*(1+1e-14))
    actual=integrate.quad(lambda t:(m-.1)*expit(-5*(t+6))*stats.norm.pdf(t),-15,15,
                          epsabs=1e-16,points=[-6,-5,-4])[0]
    per_person_bound=.8*math.exp(-17.5)
    assert 0<actual<per_person_bound
    assert .05+3000*per_person_bound<.051
    # The null uses a=5,b=-6,c=.1,d=m. Nothing about production MAP is changed.
    assert .1<=5<=5 and -6<=-6<=6 and 0<.1<.5 and .5<m<1
