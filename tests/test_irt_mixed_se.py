"""B12.6 safety gates must fail closed before independent SE validation."""
import numpy as np
import pandas as pd
import pytest
from core.irt_mixed import MixedDesign
from core.irt_mixed_se import observed_information

def fixture():
    d=MixedDesign(["1PL","2PL","3PL","4PL"])
    x=pd.DataFrame(np.tile([0.,1.,0.,1.],(40,1)),columns=list("ABCD"))
    fit={"design":d,"n":40,"converged":True,"map":False,"bound":[],
         "params":np.zeros(d.size),"quadrature":"mirt"}
    return x,fit

def test_withholds_boundary_wald_se():
    x,f=fixture();f["bound"]=[("c","D",-1)]
    r=observed_information(x,None,f)
    assert not r["valid"] and r["se"] is None and "Boundary" in r["reason"]

def test_withholds_nonconverged_se():
    x,f=fixture();f["converged"]=False
    assert not observed_information(x,None,f)["valid"]

def test_withholds_map_se():
    x,f=fixture();f["map"]=True
    assert not observed_information(x,None,f)["valid"]

def test_rejects_misaligned_sample():
    x,f=fixture();f["n"]=41
    with pytest.raises(ValueError,match="sizes"):
        observed_information(x,None,f)

def test_rejects_invalid_step():
    x,f=fixture()
    with pytest.raises(ValueError,match="settings"):
        observed_information(x,None,f,step=0)
