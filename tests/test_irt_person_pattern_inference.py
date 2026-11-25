"""Production integration against real native fits; no new Monte Carlo campaign."""
import io,json,zipfile,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from core.irt_python_mirt_adapter import fit_python_mirt
from core.irt_binary_persons import binary_person_abilities
from core.irt_binary_patterns import score_independent_persons,independent_person_scope
from validation.binary_certification.common import simulate

@pytest.fixture(scope="module",params=["3PL","4PL"])
def calibrated(request):
    model=request.param
    x=simulate(model,3000,12,[202611251101,int(model[0]),0,0])
    out=fit_python_mirt(x,model=model,estimation="MAP",n_quadpts=61)
    assert out.fit.converged
    ids=[f"C{i:04d}" for i in range(len(x))]
    people=np.array([[0]*12,[1]*12,[0,1]*6])
    return model,x,out,ids,people


def test_real_native_scores_unchanged_and_independent_intervals(calibrated):
    model,x,out,ids,held=calibrated
    before=out.fit.coef().copy(deep=True)
    original=binary_person_abilities(out,held,["H0","H1","H2"])
    result=score_independent_persons(out,held,["H0","H1","H2"],ids,True)
    np.testing.assert_array_equal(result["Yetenek (θ)"],original["Yetenek (θ)"])
    np.testing.assert_array_equal(result["Posterior SS (θ)"],original["Posterior SS (θ)"])
    assert original["Bilimsel durum"].str.contains("NOT CERTIFIED").all()
    assert result["Bilimsel durum"].str.contains("SCOPED CERTIFIED").all()
    assert result.Birey.tolist()==["H0","H1","H2"]
    assert (result["Posterior %95 alt"]<result["Posterior %95 üst"]).all()
    assert "Bayesçi" in result["Posterior aralık türü"].iloc[0]
    pd.testing.assert_frame_equal(out.fit.coef(),before)


@pytest.mark.parametrize("bad",["undeclared","overlap","wrong_scope"])
def test_independence_and_scope_fail_closed(calibrated,bad):
    from copy import copy
    model,x,out,ids,held=calibrated
    check=copy(out);labels=["H0","H1","H2"];declared=True
    if bad=="undeclared":declared=False
    if bad=="overlap":labels[0]=ids[0]
    if bad=="wrong_scope":check.n_persons=2999
    with pytest.raises(ValueError):score_independent_persons(check,held,labels,ids,declared)


def test_shared_exports_and_reproducibility_asset(calibrated,monkeypatch):
    from core.irt_python_mirt_report import build_report
    from reports import irt_binary_report as report
    from core.reproducible import runtime_hashes
    model,x,out,ids,held=calibrated
    people=score_independent_persons(out,held,["H0","H1","H2"],ids,True)
    r=build_report(out,model=model,items=[f"M{i+1}" for i in range(12)])
    monkeypatch.setattr(report,"png_bytes",lambda *a,**kw:b"\x89PNG\r\n\x1a\n")
    excel,archive=report.bundle(r,people,None,"P1 NOT CERTIFIED")
    table=pd.read_excel(io.BytesIO(excel),sheet_name="Kisi_Yetenekleri")
    assert "Posterior %95 alt" in table and "Skor GA %95 üst" in table
    assert table.Birey.tolist()==["H0","H1","H2"]
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        method=z.read(f"{model}_yontem.txt").decode()
        assert "Bayesçi" in method and "21 nokta" in method and "J30" in method
        assert "P1 NOT CERTIFIED" in method
        data=pd.read_csv(z.open("tablolar/Kisi_Yetenekleri.csv"))
        assert len(data)==3 and "Skor GA %95 alt" in data
    hashes=runtime_hashes([{"module":"core.irt_binary_patterns","function":"score_independent_persons"}])
    assert "validation/evidence/irt_closure/P2_CERTIFICATION.json" in hashes


def test_ui_cache_content_change_and_download_suppression(calibrated,monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from core.irt_python_mirt_report import build_report
    from reports import irt_binary_report as reporting
    model,x,out,ids,held=calibrated
    columns=[f"M{i+1}" for i in range(12)]
    training=pd.DataFrame(x,columns=columns);training["ID"]=ids
    scoring=pd.DataFrame(held,columns=columns);scoring["ID"]=["H0","H1","H2"]
    payload={"data":scoring.to_csv(index=False).encode()}
    class Upload:
        name="held.csv"
        def getvalue(self):return payload["data"]
    original=st.file_uploader
    monkeypatch.setattr(st,"file_uploader",lambda *a,**kw:Upload() if kw.get("key")==f"python_mirt_holdout_file_{model}" else original(*a,**kw))
    monkeypatch.setattr(reporting,"png_bytes",lambda *a,**kw:b"\x89PNG\r\n\x1a\n")
    script="""
import streamlit as st
from ui.irt_python_library_view import _independent_person_view
_independent_person_view(st.session_state['saved'],st.session_state['report'],st.session_state['data'],
                         st.session_state['columns'],'ID',st.session_state['model'])
"""
    at=AppTest.from_string(script,default_timeout=45)
    at.session_state["saved"]={"key":"calibration-v1","outcome":out}
    at.session_state["report"]=build_report(out,model=model,items=columns)
    at.session_state["data"]=training;at.session_state["columns"]=columns;at.session_state["model"]=model
    at.run();assert not at.exception and not at.get("download_button")
    next(c for c in at.checkbox if c.key==f"python_mirt_holdout_declared_{model}").set_value(True).run()
    next(b for b in at.button if b.key==f"python_mirt_holdout_run_{model}").click().run()
    assert not at.exception
    assert {d.key for d in at.get("download_button")}=={f"python_mirt_holdout_{kind}_{model}" for kind in ("csv","xlsx","zip")}
    at.run();assert len(at.get("download_button"))==3
    scoring.iloc[0,0]=1-scoring.iloc[0,0]
    payload["data"]=scoring.to_csv(index=False).encode()
    at.run();assert not at.get("download_button")
    assert f"python_mirt_holdout_result_{model}" not in at.session_state


def test_original_parameter_gates_preserved():
    from core.irt_binary_inference import EVIDENCE_34
    for q in ("b","c"):assert EVIDENCE_34[("3PL","MAP",q)][2] is False
    for q in "abcd":assert EVIDENCE_34[("4PL","MAP",q)][2] is False
    path=Path(__file__).resolve().parents[1]/"validation/evidence/binary_certification/CERTIFICATION.json"
    content=path.read_bytes()
    assert hashlib.sha1(b"blob "+str(len(content)).encode()+b"\0"+content).hexdigest()=="49b5aa914c8ea99ec319ded58bf09ac6b85a4de2"
