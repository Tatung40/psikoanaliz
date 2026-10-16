import numpy as np
import pandas as pd
from core.warning_explanations import boundary_report, opg_explanation, GUIDE


def test_boundary_report_names_and_limits():
    table=pd.DataFrame({
        "Madde":["M1","M2"], "a":[3.99,1.2], "b":[0.0,3.98],
        "c":[.34,.1], "d":[.66,.99]})
    report=boundary_report(table,"4PL",a_upper=4.,d_lower=.65)
    rows={(r["Madde"],r["Parametre"],r["Yaklaşılan sınır"])
          for _,r in report.iterrows()}
    assert ("M1","a","üst") in rows
    assert ("M1","c","üst") in rows
    assert ("M1","d","alt") in rows
    assert ("M2","b","üst") in rows
    assert set(["Kestirim","İzin verilen alt","İzin verilen üst"]).issubset(report.columns)


def test_3pl_explanation_uses_real_parameter_count():
    text=opg_explanation("OPG için parametre sayısı fazla veya örneklem yetersiz.",
                         200,25,"3PL")
    assert "75 parametre" in text
    assert "375" in text
    assert "N=200" in text
    assert "evrensel" in text


def test_default_model_does_not_flag_fixed_parameters():
    table=pd.DataFrame({"Madde":["M1"],"a":[1.],"b":[0.],
                        "c":[0.],"d":[1.]})
    assert boundary_report(table,"1PL").empty
    assert "yetenek" in GUIDE
