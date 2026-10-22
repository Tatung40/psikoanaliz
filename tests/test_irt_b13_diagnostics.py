"""B13 inference gates must not emit untrusted standard errors."""
import numpy as np
import pandas as pd
from core.irt_b13_diagnostics import calculate_diagnostics

def test_nonconverged_inference_is_withheld():
    x=pd.DataFrame(np.tile([0,1,0,1,1],(40,1)),columns=list("ABCDE"))
    items=pd.DataFrame({"Madde":list("ABCDE"),"a":[1.]*5,"b":[-.5,0,.5,1,1.5],
                        "c":[0.]*5,"d":[1.]*5})
    r=calculate_diagnostics(x,{"converged":False},items,"2PL")
    assert r["se"] is None and r["item_fit"] is None
    assert any("bastırıldı" in m for m in r["messages"])

def test_rasch_uses_existing_native_fit():
    x=pd.DataFrame(np.ones((35,5)),columns=list("ABCDE"))
    r=calculate_diagnostics(x,{"converged":True},pd.DataFrame(),"Rasch")
    assert r["se"] is None and r["item_fit"] is None
