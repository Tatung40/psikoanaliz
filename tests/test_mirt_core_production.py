"""Actual native software fixtures and real exports; NOT a new scientific campaign."""
import io,json,time,zipfile
from pathlib import Path
from functools import wraps
import numpy as np
import pandas as pd
import pytest
from core.irt_multidim import fit,report
from core.irt_mirt_production import decorate_native,refresh_policy
from reports.irt_multidim_report import downloads
from validation.mirt_stage1.compare import inputs

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"validation-output/mirt-core/software"

@pytest.fixture(scope="module")
def actual(tmp_path_factory):
    results={'_recorders':{}}
    from core import reproducible as R
    for name in ("2PL_2D","GRM_2D","GPCM_3D","PCM_2D","Rasch_2D","NRM_2D"):
        cfg,data,_,_=inputs(name,tmp_path_factory.mktemp(name))
        data=data if name=="2PL_2D" else data.head(100).copy()
        ids=[f"PERSON-{n}" for n in range(len(data))]
        recorder=R.begin('Çok Boyutlu MTK (MIRT)',globals())
        t=time.perf_counter()
        outcome=fit(data,cfg["model"],cfg["pattern"],correlated=cfg["correlated"],
                    n_quadpts=15 if name=="2PL_2D" else 5,max_iter=1000,
                    tol=1e-7 if name=="2PL_2D" else 1e-5)
        fitted=time.perf_counter();base=report(outcome,ids);result=decorate_native(outcome,base,ids)
        measured=time.perf_counter();recorder=R.finish();results['_recorders'][name]=recorder
        results[name]=(cfg,data,outcome,base,result,ids)
        folder=OUT/name;folder.mkdir(parents=True,exist_ok=True)
        (folder/"runtime.json").write_text(json.dumps(dict(model=cfg["model"],N=len(data),dimensions=cfg["q"],
            fit_seconds=fitted-t,report_seconds=measured-fitted,converged=bool(outcome["fit"].converged),
            note="One software fixture timing only; no P95 or independent scientific evidence"),indent=2))
    return results

@pytest.mark.parametrize("name",["2PL_2D","GRM_2D","GPCM_3D","PCM_2D","Rasch_2D","NRM_2D"])
def test_actual_native_tables_and_exports_preserve_values_and_withhold_item_inference(actual,name):
    cfg,data,outcome,base,result,ids=actual[name]
    for key,frame in base["tables"].items():
        if key not in ("Summary","Boundaries"):pd.testing.assert_frame_equal(result["tables"][key],frame)
    from core.irt_mirt_diagnostics import native_boundaries
    pd.testing.assert_frame_equal(result["tables"]["Boundaries"],native_boundaries(outcome["fit"]))
    assert result["metadata"]["Inferential_status"]=="WITHHELD"
    assert result["metadata"]["Item_parameter_SE"]==result["metadata"]["Item_parameter_CI"]=="WITHHELD"
    assert "universal" in result["metadata"]["Threshold_parameterization"]
    if not outcome["fit"].converged:
        assert not {"EAP","MAP","PosteriorCovariance","Curves"}&result["tables"].keys()
    else:
        assert result["tables"]["EAP"].ID.tolist()==ids
        assert "PosteriorCovariance" in result["tables"],result["metadata"]
        q=cfg["q"];cov=result["tables"]["PosteriorCovariance"]
        matrices=cov.Covariance.to_numpy().reshape(len(data),q,q)
        sd=result["tables"]["EAP"].filter(regex="^Posterior_SD_").to_numpy()
        np.testing.assert_allclose(np.diagonal(matrices,axis1=1,axis2=2),sd**2,atol=1e-10,rtol=0)
        assert np.linalg.eigvalsh(matrices).min()>=-1e-10
    if name=="2PL_2D":
        assert outcome["fit"].converged
        assert result["metadata"]["Status"]=="NUMERICALLY CERTIFIED / SCOPED"
    elif name=="NRM_2D":assert result["metadata"]["Status"]=="EXPERIMENTAL / NOT CERTIFIED"
    folder=OUT/name;items=list(data.columns)
    bundle=downloads(result,items[0] if outcome["fit"].converged else None,result["metadata"]["Labels"][0])
    for ext,payload in bundle.items():
        if payload is not None:(folder/("results."+("md" if ext=="report" else ext))).write_bytes(payload)
    frame=pd.read_csv(io.BytesIO(bundle["csv"]))
    assert frame["Report_Status"].eq(result["metadata"]["Status"]).all()
    assert frame["Report_Inferential_Status"].eq("WITHHELD").all()
    native=result["tables"]["Parameters"]
    for col in native.select_dtypes(include=[np.number]):
        np.testing.assert_allclose(frame[col],native[col],rtol=1e-14,atol=1e-14)
    book=pd.ExcelFile(io.BytesIO(bundle["xlsx"]))
    assert {"Summary","ScientificScope","Output_scope"}<=set(book.sheet_names)
    summary=pd.read_excel(book,sheet_name="ScientificScope").set_index("Setting").Value
    assert summary["Status"]==result["metadata"]["Status"] and summary["Inferential_status"]=="WITHHELD"
    with zipfile.ZipFile(io.BytesIO(bundle["zip"])) as z:
        assert json.loads(z.read("metadata.json"))["Status"]==result["metadata"]["Status"]
        assert "WITHHELD" in z.read("report.md").decode()
        assert ("surfaces.png" in z.namelist())==bool(outcome["fit"].converged)

def test_real_word_and_python_scope_cache_refresh_without_refitting(actual,monkeypatch,tmp_path):
    import core.irt_mirt_policy as policy
    import ui.irt_multidim_view as view
    from core.reproducible import recorded
    from docx import Document
    from streamlit.testing.v1 import AppTest
    cfg,data,outcome,base,result,ids=actual["2PL_2D"]
    frame=data.copy();frame.insert(0,"PersonID",ids)
    calls=[]
    @wraps(recorded(view.fit),updated=())
    def existing(observed,model,pattern,**settings):
        pd.testing.assert_frame_equal(observed,data)
        assert model=="2PL" and pattern==cfg["pattern"] and settings["correlated"]==cfg["correlated"]
        calls.append(1);return outcome
    monkeypatch.setattr(view,"fit",existing)
    page="Çok Boyutlu MTK (MIRT)"
    at=AppTest.from_file(str(ROOT/"app.py"),default_timeout=240)
    for k,v in dict(raw_df=frame,scored_df=None,item_cols=list(data.columns),id_col="PersonID",nav_page=page).items():at.session_state[k]=v
    at.run();assert not at.exception
    at.selectbox(key="md_profile").select("2PL_2D").run();assert not at.exception
    at.button(key="md_run").click().run();assert not at.exception
    saved=at.session_state[view.KEY];assert len(calls)==1
    assert saved["result"]["metadata"]["Status"]=="NUMERICALLY CERTIFIED / SCOPED"
    h=at.session_state["_academic_history"][page]
    raw=h["docx"];doc=Document(io.BytesIO(raw))
    text="\n".join([p.text for p in doc.paragraphs]+[c.text for t in doc.tables for row in t.rows for c in row.cells])
    assert "NUMERICALLY CERTIFIED" in text and "WITHHELD" in text
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"actual_m2pl_academic.docx").write_bytes(raw)
    initial=saved["bundle"]["zip"]
    at.run();assert not at.exception
    assert len(calls)==1 and at.session_state[view.KEY]["bundle"]["zip"] is initial
    assert at.session_state["_academic_history"][page]["docx"] is raw
    # Display-only curve selection changes report/export, never calibration.
    at.selectbox(key="md_plot_item").select(list(data.columns)[1]).run();assert not at.exception
    assert len(calls)==1
    assert at.session_state["_academic_history"][page]["docx"] is not raw
    # Withdraw evidence, preserving the declarative profile/settings and existing fit.
    monkeypatch.setattr(policy,"ROOT",tmp_path)
    at.run();assert not at.exception and len(calls)==1
    s=at.session_state[view.KEY];assert s["result"]["metadata"]["Status"]=="EXPERIMENTAL / NOT CERTIFIED"
    assert json.loads(zipfile.ZipFile(io.BytesIO(s["bundle"]["zip"])).read("metadata.json"))["Status"]=="EXPERIMENTAL / NOT CERTIFIED"
    changed=frame.copy();changed.iloc[0,1]=1-changed.iloc[0,1];at.session_state["raw_df"]=changed;at.run()
    assert not at.exception and view.KEY not in at.session_state

@pytest.mark.parametrize("name",["2PL_2D","GRM_2D","GPCM_3D","PCM_2D","Rasch_2D","NRM_2D"])
def test_actual_shared_docx_for_each_native_family(actual,name):
    from core.academic_screen_results import adapt_screen,validate_screen_result,convergence
    from reports.academic_docx import build_academic_docx
    from docx import Document
    cfg,data,outcome,base,result,ids=actual[name]
    recorder=actual["_recorders"][name]
    if not outcome["fit"].converged:
        assert convergence(recorder) is False
        assert not {"EAP","MAP","Curves"}&result["tables"].keys()
        return  # Verified refusal, not a scientific PASS or hidden skipped fit.
    envelope=adapt_screen("Çok Boyutlu MTK (MIRT)",recorder,list(result["tables"].items()),[],
                          [result["metadata"]["Status"],result["metadata"]["Limitations"],"WITHHELD: item SE/CI/p"],data)
    payload=build_academic_docx(envelope,validator=validate_screen_result,title=cfg["model"]+" Çok Boyutlu MTK Akademik Raporu",apa7=True)
    parsed=Document(io.BytesIO(payload))
    text="\n".join([p.text for p in parsed.paragraphs]+[c.text for table in parsed.tables for row in table.rows for c in row.cells])
    assert parsed.tables and "WITHHELD" in text and "Item" in text
    (OUT/name/"actual_academic.docx").write_bytes(payload)

@pytest.mark.parametrize("profile",["2PL_2D","2PL_3D","GRM_2D","GRM_3D","GPCM_3D","PCM_2D","Rasch_2D"])
def test_every_registered_preset_sets_exact_spec_without_fitting(profile,tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    import ui.irt_multidim_view as view
    cfg,data,_,_=inputs(profile,tmp_path/profile)
    def forbidden(*args,**kwargs):raise AssertionError("Preset selection must not fit")
    monkeypatch.setattr(view,"fit",forbidden)
    at=AppTest.from_file(str(ROOT/"app.py"),default_timeout=120)
    for k,v in dict(raw_df=data,scored_df=None,item_cols=list(data.columns),id_col=None,
                    nav_page="Çok Boyutlu MTK (MIRT)").items():at.session_state[k]=v
    at.run();assert not at.exception
    at.selectbox(key="md_model").select(cfg["model"]).run();assert not at.exception
    at.selectbox(key="md_profile").select(profile).run();assert not at.exception
    assert at.selectbox(key="md_dims").value==cfg["q"]
    assert at.selectbox(key="md_corr").value=={False:"Ortogonal",True:"Yalnız F1–F2","full":"Tümü"}[cfg["correlated"]]
    assert at.selectbox(key=f"md_quad_{cfg['q']}").value==15
    assert at.number_input(key="md_profile_iter").value==1000
    assert at.number_input(key="md_profile_tol").value==1e-7
    assert any(b.key=="md_run" for b in at.button)
    assert view.KEY not in at.session_state
