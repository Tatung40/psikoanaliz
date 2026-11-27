"""P4-C1 production integration: shared table labels, fail-closed scope, XLSX/ZIP identical to the screen table."""
import io, sys, zipfile
import numpy as np
import pandas as pd
import pytest
from core.irt_python_mirt_adapter import fit_python_mirt
from core.polytomous_inference import parameter_table
sys.path.insert(0, "validation")
import polytomous_mc_study as S

ITEMS = [f"m{j}" for j in range(S.K)]


@pytest.fixture(scope="module")
def fits():
    out = {}
    for m in ("GRM", "PCM", "GPCM", "NRM"):
        x, _ = S.simulate(m, 2000, 777000 + len(out))
        out[m] = fit_python_mirt(x, model=m, n_categories=S.C)
    return out


def test_seven_blocks_certified_in_scope_and_nrm_oakes_recorded(fits):
    blocks = set()
    for m, o in fits.items():
        t, reason = parameter_table(o, ITEMS); assert reason is None
        assert t["Bilimsel durum (GA)"].str.startswith("SCOPED CERTIFIED").all(), m
        blocks |= {(m, b) for b in t["Parametre"].unique()}
    assert fits["NRM"].fit.se_method == "oakes"   # library-recorded metadata
    assert blocks == {("GRM", "discrimination"), ("GRM", "thresholds"), ("PCM", "steps"), ("GPCM", "discrimination"),
                      ("GPCM", "steps"), ("NRM", "slopes"), ("NRM", "intercepts")}


def test_fail_closed_outside_scope(fits):
    import dataclasses
    import copy
    nose = copy.copy(fits["NRM"].fit); nose.se_method = None
    for change in ({"n_persons": 500}, {"fit": nose}, {"solver_settings": {"max_iter": 500, "tol": 1e-4, "n_quadpts": 61}},
                   {"estimation": "MAP"}, {"n_items": 12}):
        o = dataclasses.replace(fits["NRM"], **change)
        assert (parameter_table(o, ITEMS + ["x", "y"])[0]["Bilimsel durum (GA)"] == "NOT CERTIFIED").all(), change
    assert parameter_table(dataclasses.replace(fits["GRM"], model="RSM"), ITEMS)[0] is None


def test_xlsx_and_zip_carry_identical_table(fits):
    from core.polytomous_inference import export_bundle
    o = fits["GPCM"]; t, _ = parameter_table(o, ITEMS)
    excel, archive = export_bundle(t)
    sheet = pd.read_excel(io.BytesIO(excel), sheet_name="Parametre_SH_GA")
    assert sheet["Bilimsel durum (GA)"].tolist() == t["Bilimsel durum (GA)"].tolist()
    assert np.allclose(sheet["%95 GA alt"], t["%95 GA alt"])
    z = zipfile.ZipFile(io.BytesIO(archive)); assert pd.read_csv(z.open("Parametre_SH_GA.csv"))["Bilimsel durum (GA)"].tolist() == t["Bilimsel durum (GA)"].tolist()

def test_uncertified_intervals_and_se_never_leak_to_exports(fits):
    import dataclasses
    from core.polytomous_inference import export_bundle
    for model,outcome in fits.items():
        unknown=dataclasses.replace(outcome,n_persons=500)
        table,reason=parameter_table(unknown,ITEMS);assert reason is None
        assert table[["SH","%95 GA alt","%95 GA üst"]].isna().all().all()
        excel,archive=export_bundle(table)
        sheet=pd.read_excel(io.BytesIO(excel),sheet_name="Parametre_SH_GA")
        assert sheet[["SH","%95 GA alt","%95 GA üst"]].isna().all().all()
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            csv=pd.read_csv(z.open("Parametre_SH_GA.csv"))
            assert csv[["SH","%95 GA alt","%95 GA üst"]].isna().all().all()
        assert np.isfinite(table["Kestirim"]).all()

def test_cached_polytomous_outputs_follow_parameters_se_settings_and_evidence(fits,monkeypatch):
    import dataclasses,copy
    from core.polytomous_inference import _cache_key
    import core.polytomous_certification as C
    out=fits["GPCM"];a=_cache_key(out,ITEMS)
    assert _cache_key(out,ITEMS)==a
    assert _cache_key(dataclasses.replace(out,n_persons=500),ITEMS)!=a
    assert _cache_key(dataclasses.replace(out,solver_settings=dict(out.solver_settings,n_quadpts=61)),ITEMS)!=a
    from types import SimpleNamespace
    changed=SimpleNamespace(model=out.fit.model,standard_errors={k:np.asarray(v).copy() for k,v in out.fit.standard_errors.items()},
                            se_method=out.fit.se_method,converged=True)
    next(iter(changed.standard_errors.values())).flat[0]*=1.1
    assert _cache_key(dataclasses.replace(out,fit=changed),ITEMS)!=a
    monkeypatch.setattr(C,"_evidence_fingerprint",lambda:"withdrawn")
    assert _cache_key(out,ITEMS)!=a

@pytest.mark.parametrize("model",["GRM","PCM","GPCM","NRM"])
def test_polytomous_word_numeric_parity_cache_and_withdrawal(fits,model,monkeypatch,tmp_path):
    import json
    from functools import wraps
    from pathlib import Path
    from docx import Document
    from streamlit.testing.v1 import AppTest
    import ui.irt_python_library_view as view
    import core.polytomous_certification as C
    from core.reproducible import recorded
    idx=("GRM","PCM","GPCM","NRM").index(model)
    x,_=S.simulate(model,2000,777000+idx)
    frame=pd.DataFrame(x,columns=ITEMS);out=fits[model];calls=[]
    @wraps(recorded(view.fit_python_mirt),updated=())
    def fitted(data,**kwargs):
        np.testing.assert_array_equal(data,x)
        assert kwargs["model"]==model and kwargs["estimation"]=="ML"
        calls.append(1);return out
    monkeypatch.setattr(view,"fit_python_mirt",fitted)
    page="MTK: Python mirt "+model
    at=AppTest.from_file(str(Path(__file__).resolve().parents[1]/"app.py"),default_timeout=180)
    for key,value in dict(raw_df=frame,scored_df=frame%2,item_cols=ITEMS,id_col=None,group_col=None,is_dichotomous=False,nav_page=page).items():at.session_state[key]=value
    at.run();at.button(key="python_mirt_start_"+model).click().run();assert not at.exception
    expected,_=parameter_table(out,ITEMS)
    result=at.session_state["_academic_history"][page]
    candidate=next(rows for rows in result["result"]["tables"].values() if rows and "%95 GA alt" in rows[0])
    actual=pd.DataFrame(candidate)
    for col in ("Kestirim","SH","%95 GA alt","%95 GA üst"):
        np.testing.assert_array_equal(actual[col].to_numpy(float),expected[col].to_numpy(float))
    raw=result["docx"];doc=Document(io.BytesIO(raw))
    assert any("P4-C1" in cell.text for table in doc.tables for row in table.rows for cell in row.cells)
    at.run();assert not at.exception
    assert at.session_state["_academic_history"][page]["docx"] is raw and len(calls)==1
    evidence=json.loads(C.REPORT.read_text())
    for cell in evidence.values():cell["CERTIFIED"]=False
    target=tmp_path/"withdrawn.json";target.write_text(json.dumps(evidence));monkeypatch.setattr(C,"REPORT",target)
    at.run();assert not at.exception and len(calls)==1
    table=at.session_state["python_mirt_poly_inference"][model]["table"]
    assert table[["SH","%95 GA alt","%95 GA üst"]].isna().all().all()
    assert at.session_state["_academic_history"][page]["docx"] is not raw
