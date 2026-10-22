import io
import zipfile
import pandas as pd
import numpy as np
from reports.irt_b14_narrative import guided_report
from reports.irt_b14_report import export_zip

def case(model="NRM"):
    data=pd.DataFrame((np.arange(80)[:,None]+np.arange(5)[None,:])%5,columns=list("ABCDE"))
    return {"model":model,"engine":"Python mirt 1.2.0","N":80,"k":5,
        "labels":tuple("ABCDE"),"coded":data,
        "result":type("Stub",(),{"coef_with_se":lambda self:pd.DataFrame({"estimate":[.3]})})()}

def test_nominal_narrative_never_claims_ordered_choices():
    text=guided_report(case())
    assert "sıralı başarı düzeyi kabul etmez" in text
    assert "bağımsız" in text
    assert "80" in text

def test_sparse_frequency_is_described():
    obj=case("GRM")
    obj["coded"].loc[:77,"A"]=0
    text=guided_report(obj)
    assert "5'in altındadır" in text

def test_zip_includes_explanatory_report():
    data=export_zip(case())
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert "B14_Aciklamali_Rapor.md" in z.namelist()
        assert "keşifsel" in z.read("B14_Aciklamali_Rapor.md").decode("utf-8")
