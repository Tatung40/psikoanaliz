import numpy as np
import pandas as pd
from validation.r_mirt_phase1.m12_diagnostic import scores
from validation.r_mirt_phase1.equivalence_audit import evaluate


def test_physical_scores_match_independent_likelihood_differences():
    x=np.array([[0,0],[0,1],[1,0],[1,1],[1,1]])
    p=pd.DataFrame(dict(a=[1.2,2.],b=[-.4,.7],c=[.1,.2],d=[.85,.9]),index=['M1','M12'])
    theta=np.array([-2.,-1.,0.,1.,2.]);weights=np.array([.05,.2,.5,.2,.05])
    gradient=scores(x,p,theta,weights)
    for item in p.index:
        for name in p.columns:
            plus,minus=p.copy(),p.copy()
            plus.loc[item,name]+=1e-5;minus.loc[item,name]-=1e-5
            numerical=(evaluate(x,plus,theta,weights)[2]-evaluate(x,minus,theta,weights)[2])/2e-5
            np.testing.assert_allclose(gradient.loc[item,name],numerical,rtol=1e-7,atol=1e-8)


def test_box_scores_ignore_blocked_directions_but_keep_feasible_ascent():
    from validation.r_mirt_phase1.m12_diagnostic import feasible_scores
    p=pd.DataFrame(dict(a=[.1,5.,1.],b=[-6.,6.,0.],c=[0.,.5,.2],d=[.5,1.,.7]))
    g=pd.DataFrame(dict(a=[-2.,2.,1.],b=[2.,-2.,1.],c=[-3.,3.,.4],d=[4.,-4.,.6]))
    result=feasible_scores(p,g)
    np.testing.assert_array_equal(result.a,[0.,0.,1.])
    np.testing.assert_array_equal(result.c,[0.,0.,.4])
    np.testing.assert_array_equal(result.b,g.b)
    np.testing.assert_array_equal(result.d,g.d)
    assert g.a.iloc[0]==-2.  # Diagnostic projection never alters the stored gradient.
