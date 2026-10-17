import pandas as pd
from core.irt_model_reliability import model_reliability

def _fit(converged=True, a=1.2, c=.12, d=.95, k=5, convergence=None):
    # 4PL with k=5 is structurally identifiable (20 <= 2**5-1).
    items=pd.DataFrame({"Madde":["M1","M2","M3","M4","M5"],
                        "a":[a,1.,1.5,1.1,1.3],"b":[-.3,0.,.7,.2,-.5],
                        "c":[c,.1,.13,.11,.12],"d":[d,.96,.92,.94,.93]}).head(k)
    out={"model":"4PL","k":k,"converged":converged,
         "a_upper":4.,"d_lower":.65,"items":items}
    if convergence is not None:
        out["convergence"]=convergence
    return out

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


def test_structurally_unidentified_fit_is_labelled_as_such():
    x=model_reliability(_fit(k=4))
    assert x["Karar"]=="Yapısal olarak tanımlanamaz"
    assert (x["Serbest parametre"],x["Örüntü serbestlik derecesi"])==(16,15)


def test_loglik_only_convergence_is_cautioned_not_rejected():
    x=model_reliability(_fit(convergence={"parameter_criterion_met":False,
                                          "message":"parametreler hâlâ değişiyor"}))
    assert x["Yakınsama"] and not x["Parametre yakınsaması"]
    assert x["Karar"]=="İhtiyatla yorumlanmalı"
    assert "parametreler hâlâ değişiyor" in x["Açıklamalar"]
