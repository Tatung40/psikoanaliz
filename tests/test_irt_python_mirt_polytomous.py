"""Fast integration smoke tests against real pinned Python mirt.

Do not confuse successful execution with independent psychometric validation.
"""
import numpy as np
import pytest
from core.irt_python_mirt_adapter import fit_python_mirt

@pytest.mark.parametrize("model",["PCM","GPCM","RSM","NRM"])
def test_polytomous_fit_model(model):
    import mirt
    x=mirt.simdata(model=("PCM" if model=="RSM" else model),n_persons=120,n_items=5,n_categories=4,seed=20261022)
    x=np.asarray(x)
    assert x.ndim==2 and x.shape==(120,5)
    r=fit_python_mirt(x,model=model,n_categories=4)
    assert r.model==model and r.fit is not None
    assert r.scientific_status=="EXPERIMENTAL_NOT_INDEPENDENTLY_VALIDATED"

def test_category_guard_rejects_incomplete_category_set():
    x=np.tile([0,1,0,1,0],(50,1))
    with pytest.raises(ValueError,match="All categories"):
        fit_python_mirt(x,model="GPCM",n_categories=4)
