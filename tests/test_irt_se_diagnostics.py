import pandas as pd
from core.irt_se_diagnostics import hessian_boundary_diagnostics


def test_hessian_boundary_diagnostics_reports_exact_item_and_parameter():
    items=pd.DataFrame({
        "Madde":["M1","M2","M3"],
        "a":[4.,1.2,1.5],"b":[.4,-4.,0.],
        "c":[.1,0.,.2],"d":[.7,.9,1.]})
    tab=hessian_boundary_diagnostics(items,"4PL",a_upper=4.,d_lower=.65)
    flags={(r["Madde"],r["Parametre"],r["Sorun"])
           for _,r in tab.iterrows()}
    assert ("M1","a","Üst sınır") in flags
    assert ("M2","b","Alt sınır") in flags
    assert ("M2","c","Alt sınır") in flags
    assert ("M3","d","Üst sınır") in flags
    assert len(tab)==4


def test_fixed_parameters_are_never_misreported():
    items=pd.DataFrame({"Madde":["M1"],"a":[1.],"b":[.2],
                        "c":[0.],"d":[1.]})
    assert hessian_boundary_diagnostics(items,"1PL").empty
    assert hessian_boundary_diagnostics(items,"2PL").empty


def test_near_boundary_matches_hessian_policy():
    items=pd.DataFrame({"Madde":["M1"],"a":[3.9995],"b":[0.],
                        "c":[.2],"d":[.8]})
    tab=hessian_boundary_diagnostics(items,"3PL",a_upper=4.)
    assert list(tab["Parametre"])==["a"]
