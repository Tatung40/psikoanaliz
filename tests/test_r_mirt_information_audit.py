import numpy as np
import pandas as pd
from validation.r_mirt_phase1.information_audit import information, stability


def test_second_order_stencils_preserve_cross_curvature_at_active_bounds():
    p=pd.DataFrame(dict(a=[5.],b=[.2],c=[0.],d=[1.]),index=['M12'])
    expected=np.array([[4.,.3,.2,0.],[.3,3.,.1,.1],[.2,.1,2.,.2],[0.,.1,.2,1.]])
    score=lambda q:pd.DataFrame((-expected@q.to_numpy().ravel()).reshape(1,4),index=p.index,columns=p.columns)
    for h in (1e-4,5e-5):
        actual,stencils=information(score,p,h)
        np.testing.assert_allclose(actual,expected,rtol=1e-8,atol=1e-9)
        assert [s['stencil'] for s in stencils]==['backward','central','forward','backward']


def test_instability_and_indefiniteness_block_quadratic_eligibility():
    assert not stability(np.diag([1.,2.]),np.diag([-.1,2.]))['positive_definiteness_exceeds_observed_perturbation']
    assert not stability(np.diag([.01,2.]),np.diag([.5,2.]))['positive_definiteness_exceeds_observed_perturbation']
    assert stability(np.diag([1.,2.]),np.diag([1.+1e-8,2.]))['positive_definiteness_exceeds_observed_perturbation']
