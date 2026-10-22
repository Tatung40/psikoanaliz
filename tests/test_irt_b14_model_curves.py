import numpy as np
import pytest
from core.irt_b14_model_curves import fitted_category_curves

class FakeModel:
    def probability(self,theta):
        n=len(theta)
        return np.tile(np.array([.2,.3,.5]),(n,4,1))

class FakeFit:
    model=FakeModel()

def test_category_curve_normalization():
    result={"result":FakeFit(),"k":4,"labels":("A","B","C"),"items":("I1","I2","I3","I4")}
    frame=fitted_category_curves(result,np.linspace(-2,2,9))
    assert len(frame)==108
    assert np.allclose(frame.groupby(["Theta","Madde"])["Olasılık"].sum(),1)

def test_shape_mismatch_rejected():
    class Wrong:
        def probability(self,t):
            return np.ones((len(t),4))
    class Fit:
        model=Wrong()
    result={"result":Fit(),"k":4,"labels":("A","B","C"),"items":("I1","I2","I3","I4")}
    with pytest.raises(ValueError,match="shape"):
        fitted_category_curves(result)
