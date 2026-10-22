"""B13 curve regression tests: mathematical identities across dichotomous models."""
import numpy as np
import pandas as pd
import pytest
from core.irt_b13_curves import compute_curves,check_curves

@pytest.mark.parametrize("model,a,c,d",[
    ("Rasch",1.0,0.0,1.0),("1PL",1.0,0.0,1.0),
    ("2PL",1.5,0.0,1.0),("3PL",1.5,0.2,1.0),
    ("4PL",1.5,0.2,0.91)])
def test_information_identity(model,a,c,d):
    items=pd.DataFrame({"a":[a,1.0],"b":[-0.4,0.8],"c":[c,c],"d":[d,d]})
    z=check_curves(items,model)
    assert z["passed"],z
    v=compute_curves(items)
    assert v["icc"].shape==(161,2)
    assert v["iic"].shape==(161,2)
    assert np.isfinite(v["tif"]).all()

def test_invalid_asymptote_is_rejected():
    items=pd.DataFrame({"a":[1.0],"b":[0.0],"c":[0.8],"d":[0.4]})
    with pytest.raises(ValueError,match="bounds"):
        compute_curves(items)

def test_model_structure_is_checked():
    items=pd.DataFrame({"a":[1.0],"b":[0.0],"c":[0.2],"d":[1.0]})
    assert not check_curves(items,"2PL")["passed"]
