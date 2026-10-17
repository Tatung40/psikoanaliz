import pandas as pd
from core.irt_model_reliability import model_reliability

def _fit(converged=True, a=1.2, c=.12, d=.95):
    return {"model":"4PL","k":3,"converged":converged,
            "a_upper":4.,"d_lower":.65,
            "items":pd.DataFrame({"Madde":["M1","M2","M3"],
                                  "a":[a,1.,1.5],"b":[-.3,0.,.7],
                                  "c":[c,.1,.13],"d":[d,.96,.92]})}

def test_no_boundary_does_not_claim_validation():
    x=model_reliability(_fit())
    assert x["Yakınsama"]
    assert x["Sınırdaki parametre"]==0
    assert "bağımsız doğrulama" in x["Karar"]

def test_boundary_is_cautioned_with_count():
    x=model_reliability(_fit(a=4.,c=0.,d=1.))
    assert x["Sınırdaki madde"]==1
    assert x["Sınırdaki parametre"]==3
    assert x["Karar"]=="İhtiyatla yorumlanmalı"

def test_nonconverged_is_not_reported_as_reliable():
    x=model_reliability(_fit(converged=False))
    assert x["Karar"]=="Güvenilir kestirim sağlanamadı"
