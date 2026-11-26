"""Real native calibration/UI/export regression for the separately gated P3 reference."""
import io,json,os,subprocess,sys,time,zipfile
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from docx import Document
from streamlit.testing.v1 import AppTest
from validation.binary_certification.common import simulate
from core.irt_4pl_itemfit import reference_table,P_COLUMN,STATUS_COLUMN
ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def calibrated():
    x=simulate("4PL",3000,12,[202611264901,4,0,0])
    data=pd.DataFrame(x,columns=[f"I{j+1:02d}" for j in range(12)])
    at=AppTest.from_file(str(ROOT/"app.py"),default_timeout=240)
    page="MTK: Python mirt 4PL"
    for k,v in dict(raw_df=data,scored_df=data.copy(),item_cols=list(data),id_col=None,group_col=None,
                    is_dichotomous=True,nav_page=page).items():at.session_state[k]=v
    at.run();assert not at.exception
    at.checkbox(key="python_mirt_solver_advanced_4PL").check().run()
    assert at.number_input(key="python_mirt_solver_4PL_max_iter").value==50000
    assert at.number_input(key="python_mirt_solver_4PL_tol").value==1e-10
    at.number_input(key="python_mirt_solver_4PL_max_iter").set_value(500)
    at.number_input(key="python_mirt_solver_4PL_tol").set_value(1e-4)
    at.run()
    at.checkbox(key="python_mirt_output_4PL_persons").uncheck().run()
    at.button(key="python_mirt_start_4PL").click().run()
    assert not at.exception
    saved=at.session_state["python_mirt_fit_4PL"]
    assert saved["outcome"].fit.converged
    assert saved["outcome"].solver_settings==dict(max_iter=500,tol=1e-4,n_quadpts=61)
    frame=saved["p3_itemfit"]["frame"]
    return SimpleNamespace(at=at,data=data,saved=saved,outcome=saved["outcome"],frame=frame,page=page)


def test_real_native_reference_keeps_ordinary_chi_square_closed(calibrated):
    c=calibrated
    assert c.frame[P_COLUMN].notna().all()
    assert c.frame["p_value"].isna().all()
    assert c.frame[STATUS_COLUMN].str.startswith("SCOPED CERTIFIED").all()
    assert np.all((c.frame[P_COLUMN]*4096)%1==0)
    co=c.outcome.fit.coef().copy(deep=True)
    result,reason=reference_table(c.outcome,c.data.to_numpy(),c.saved["p3_reference_seed"])
    assert reason is None and np.array_equal(result[P_COLUMN],c.frame[P_COLUMN])
    pd.testing.assert_frame_equal(co,c.outcome.fit.coef())
    print("P3_PUBLIC_REFERENCE_SECONDS",result.attrs["p3_4pl_reference"]["seconds"])


@pytest.mark.parametrize("change",[dict(model="3PL"),dict(estimation="ML"),dict(n_persons=2999),
    dict(n_items=11),dict(solver_settings=dict(max_iter=50000,tol=1e-10,n_quadpts=61)),dict(solver_settings=dict(max_iter=500,tol=1e-4,n_quadpts=21))])
def test_out_of_scope_outputs_remain_suppressed(calibrated,change):
    result,reason=reference_table(replace(calibrated.outcome,**change),calibrated.data.to_numpy(),[1,2,3])
    assert result is None and reason


def test_numerical_refusal_and_changed_cohort_do_not_emit_p(calibrated,monkeypatch):
    from validation.p3_empirical import constrained
    from ui.irt_python_library_view import _itemfit
    def bad(*args,**kwargs):raise ValueError("invalid full covariance")
    monkeypatch.setattr(constrained,"analyse",bad)
    result,reason=reference_table(calibrated.outcome,calibrated.data.to_numpy(),[1,2,3])
    assert result is None and "sayısal ret" in reason
    changed=calibrated.data.copy();changed.iloc[0,0]=1-changed.iloc[0,0]
    with pytest.raises(ValueError,match="Kalibrasyon verisi değişti"):
        _itemfit(calibrated.saved,changed,"4PL")


def test_actual_csv_xlsx_zip_word_and_portable_sources(calibrated,tmp_path):
    c=calibrated
    from ui.irt_python_library_view import _csv
    roundtrip=pd.read_csv(io.BytesIO(_csv(c.frame)))
    assert np.allclose(roundtrip[P_COLUMN],c.frame[P_COLUMN],atol=1e-15)
    xlsx,raw=c.at.session_state["python_mirt_bundle"]["4PL"]["files"]
    excel=pd.read_excel(io.BytesIO(xlsx),sheet_name="Madde_Uyumu")
    assert np.allclose(excel[P_COLUMN],c.frame[P_COLUMN])
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        table=pd.read_csv(io.BytesIO(z.read("tablolar/Madde_Uyumu.csv")))
        assert np.allclose(table[P_COLUMN],c.frame[P_COLUMN])
        note=z.read("4PL_yontem.txt").decode()
        assert "4PL Q: ayrı kovaryans-standartlaştırılmış" in note and "S-X² veya olağan χ² p değeri değildir" in note
    word=c.at.session_state["_academic_history"][c.page]["docx"]
    doc=Document(io.BytesIO(word))
    text="\n".join(cell.text for table in doc.tables for row in table.rows for cell in row.cells)
    assert P_COLUMN in text and "4PL Q" in text
    from ui.reproducible_capture import replay_manifest,python_code,bundle
    manifest=replay_manifest(c.at.session_state["_repro_history"])
    assert manifest["status"]=="GENERATED / UNVERIFIED",manifest["unsupported"]
    code=python_code(manifest);compile(code,"analysis.py","exec")
    with zipfile.ZipFile(io.BytesIO(bundle(manifest,code))) as z:
        for name in ("validation/p3_empirical/constrained.py","validation/p3_empirical/kernel.py",
                     "validation/p3_sx2_analytic/kernel.py","validation/evidence/p3_4pl_closure/CERTIFICATION.json"):
            assert "runtime/"+name in z.namelist()
        z.extractall(tmp_path)
    check="import sys;sys.path.insert(0,'runtime');from core.irt_4pl_itemfit import PROOF;from validation.p3_empirical.constrained import analyse;assert PROOF.is_file()"
    run=subprocess.run([sys.executable,"-c",check],cwd=tmp_path,env=dict(os.environ,PYTHONPATH=""),
                       capture_output=True,text=True,timeout=30)
    assert run.returncode==0,run.stderr


def test_unchanged_rerun_and_graphic_change_preserve_fit_and_reference(calibrated):
    c=calibrated;at=c.at;fit=c.outcome
    before=at.session_state["_academic_history"][c.page]["docx"];seed=list(c.saved["p3_reference_seed"])
    at.run();assert not at.exception
    assert at.session_state["_academic_history"][c.page]["docx"] is before
    assert at.session_state["python_mirt_fit_4PL"]["outcome"] is fit
    selector=at.selectbox(key="python_mirt_plot_item_4PL")
    selector.set_value(selector.options[1]).run()
    assert not at.exception
    assert at.session_state["python_mirt_fit_4PL"]["outcome"] is fit
    assert at.session_state["python_mirt_fit_4PL"]["p3_reference_seed"]==seed
    assert np.array_equal(at.session_state["python_mirt_fit_4PL"]["p3_itemfit"]["frame"][P_COLUMN],c.frame[P_COLUMN])


def test_proof_withdrawal_invalidates_ui_and_all_downloads(calibrated,monkeypatch,tmp_path):
    import core.irt_4pl_itemfit as gate
    c=calibrated;at=c.at
    invalid=tmp_path/"proof.json"
    old=json.loads(gate.PROOF.read_text());old["status"]="NOT CERTIFIED"
    invalid.write_text(json.dumps(old))
    monkeypatch.setattr(gate,"PROOF",invalid)
    at.run();assert not at.exception
    current=at.session_state["python_mirt_fit_4PL"]["p3_itemfit"]["frame"]
    assert P_COLUMN not in current or current[P_COLUMN].isna().all()
    assert current["p_value"].isna().all()
    xlsx,raw=at.session_state["python_mirt_bundle"]["4PL"]["files"]
    excel=pd.read_excel(io.BytesIO(xlsx),sheet_name="Madde_Uyumu")
    assert P_COLUMN not in excel or excel[P_COLUMN].isna().all()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        csv=pd.read_csv(io.BytesIO(z.read("tablolar/Madde_Uyumu.csv")))
        assert P_COLUMN not in csv or csv[P_COLUMN].isna().all()
    doc=Document(io.BytesIO(at.session_state["_academic_history"][c.page]["docx"]))
    text="\n".join(cell.text for table in doc.tables for row in table.rows for cell in row.cells)
    assert P_COLUMN not in text


def test_frozen_confirmation_and_preserved_certificates():
    from validation.p3_4pl_closure.frozen import audit
    result=audit()
    assert result["records"]==7800 and result["no_new_fits"]
    from tests.test_p3_empirical import test_primary_registry_and_original_gates_unchanged
    test_primary_registry_and_original_gates_unchanged()


def test_actual_public_postfit_latency(calibrated):
    import time
    c=calibrated;seconds=[]
    for i in range(12):
        start=time.perf_counter()
        result,reason=reference_table(c.outcome,c.data.to_numpy(),[202611264990,4,i])
        seconds.append(time.perf_counter()-start)
        assert reason is None and result[P_COLUMN].notna().all()
    print("P3N_PUBLIC_LATENCY_JSON",json.dumps({"median":float(np.median(seconds)),
        "P95":float(np.percentile(seconds,95)),"calls":len(seconds),"N":3000,"J":12,"new_fits":0}))


def test_new_reference_seed_updates_screen_and_downloads_without_refitting(calibrated):
    c=calibrated;at=c.at
    saved=at.session_state["python_mirt_fit_4PL"];fit=saved["outcome"]
    seed=[202611264991,4,0];saved["p3_reference_seed"]=seed;saved.pop("p3_itemfit",None)
    at.session_state["python_mirt_fit_4PL"]=saved
    at.run();assert not at.exception
    current=at.session_state["python_mirt_fit_4PL"]
    assert current["outcome"] is fit
    frame=current["p3_itemfit"]["frame"]
    assert frame["Referans seed"].eq(str(seed)).all()
    xlsx,raw=at.session_state["python_mirt_bundle"]["4PL"]["files"]
    excel=pd.read_excel(io.BytesIO(xlsx),sheet_name="Madde_Uyumu")
    assert excel["Referans seed"].eq(str(seed)).all()
    assert np.array_equal(excel[P_COLUMN],frame[P_COLUMN])
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        csv=pd.read_csv(io.BytesIO(z.read("tablolar/Madde_Uyumu.csv")))
        assert csv["Referans seed"].eq(str(seed)).all()
        assert np.allclose(csv[P_COLUMN],frame[P_COLUMN],rtol=0,atol=1e-15)
    doc=Document(io.BytesIO(at.session_state["_academic_history"][c.page]["docx"]))
    text="\n".join(cell.text for table in doc.tables for row in table.rows for cell in row.cells)
    assert str(seed) in text
