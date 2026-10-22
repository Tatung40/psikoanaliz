import io
import zipfile
import numpy as np
import pandas as pd
from reports.irt_b14_audit import interpretation_audit
from reports.irt_b14_report import export_zip

def case(converged):
    coded=pd.DataFrame((np.arange(90)[:,None]+np.arange(5)[None,:])%5,columns=list("ABCDE"))
    return {"model":"NRM","engine":"Python mirt 1.2.0","status":"DENEYSEL",
        "N":90,"k":5,"labels":tuple("ABCDE"),"coded":coded,
        "result":type("Stub",(),{"converged":converged,
                                 "coef_with_se":lambda self:pd.DataFrame({"p":[.4]})})()}

def test_success_does_not_claim_scientific_fit():
    a=interpretation_audit(case(True))
    assert a["publication_ready"] is False
    assert "KEŞİFSEL" in a["decision"]
    assert "Doğrulanmadı" in set(a["table"]["Durum"])

def test_failed_convergence_blocks_inference():
    a=interpretation_audit(case(False))
    assert "ÖNCE KESTİRİM" in a["decision"]

def test_audit_included_in_archive():
    with zipfile.ZipFile(io.BytesIO(export_zip(case(True)))) as z:
        assert "bilimsel_denetim.csv" in z.namelist()
        assert "B14_rapor.xlsx" in z.namelist()
