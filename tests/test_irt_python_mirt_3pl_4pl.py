"""Real Python-mirt 3PL/4PL execution smoke tests, distinct from psychometric validation."""
import numpy as np
import pytest
from core.irt_python_mirt_adapter import fit_python_mirt

@pytest.mark.parametrize("model",["3PL","4PL"])
def test_python_mirt_dichotomous_real_calibration(model):
    import mirt
    data=mirt.simdata(model=model,n_persons=240,n_items=8,seed=20261022)
    x=np.asarray(data)
    assert x.shape==(240,8)
    assert set(np.unique(x)).issubset({0,1})
    out=fit_python_mirt(x,model=model)
    assert out.model==model
    assert out.engine=="python_mirt"
    assert out.fit is not None
    assert out.scientific_status=="EXPERIMENTAL_NOT_INDEPENDENTLY_VALIDATED"

def test_4pl_rejects_nonbinary_data():
    with pytest.raises(ValueError,match="0/1"):
        fit_python_mirt(np.full((40,8),2),model="4PL")
