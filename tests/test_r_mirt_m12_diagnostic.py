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
