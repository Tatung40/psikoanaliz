
"""Public native binary global-fit contracts; targeted, no new scientific campaign."""
import io,json,zipfile
from dataclasses import replace
from types import SimpleNamespace
import numpy as np,pandas as pd,pytest
from core.irt_global_fit import global_fit,comparison_context,nested_lrt
from core.irt_python_mirt_adapter import fit_python_mirt
from core.irt_python_mirt_report import build_report

@pytest.fixture(scope="module",params=["Rasch","1PL","2PL","3PL","4PL"])
def actual(request):
    model=request.param
    path=("tests/fixtures/irt_b11/data_irt"+model.lower()+"_ok.csv" if model in ("3PL","4PL") else "tests/fixtures/irt_b10/data_irt1pl_ok.csv")
    x=pd.read_csv(path).iloc[:300,:6].astype(int)
    out=fit_python_mirt(x.to_numpy(),model=model,estimation="MAP" if model in ("3PL","4PL") else "ML",
                        common_discrimination=model=="1PL",n_quadpts=61)
    assert out.fit.converged
    return model,x,out

def test_actual_native_global_indices_and_map_refusal(actual):
    model,x,out=actual
    value=global_fit(out,x.to_numpy(),seed=[9101,1])
    row=value["table"].iloc[0]
    if row["M2 etkin sd"]<=0:
        assert model=="4PL" and all(np.isnan(row[k]) for k in ("M2","RMSEA2 (N−1)","CFI","TLI","M2 asimptotik p"))
        assert "No residual degrees of freedom" in row["Sınırlılık"]
    else:
        assert np.isfinite(row["M2"]) and np.isfinite(row["RMSEA2 (N−1)"])
    assert np.isfinite(row["SRMSR"])
    assert value["residuals"].shape==(6,6)
    np.testing.assert_allclose(value["residuals"],value["residuals"].T,rtol=0,atol=1e-14)
    assert np.isnan(row["Küresel referans p"])    # N300/J6 is outside ALL confirmation cells
    if model in ("3PL","4PL"):
        assert np.isnan(row["M2 asimptotik p"])
        assert "DESCRIPTIVE" in row["Bilimsel durum"]
    report=build_report(out,model=model,items=x.columns,global_diagnostics=value)
    np.testing.assert_equal(report.metadata["Global · M2"],row["M2"])
    assert report.global_fit is value
    assert report.metadata["Kestirim"]==out.estimation
    with pytest.raises(ValueError,match="nonconvergence"):
        from copy import copy
        bad=copy(out.fit);bad.converged=False
        global_fit(replace(out,fit=bad),x)
    badx=x.astype(float);badx.iloc[0,0]=np.nan
    with pytest.raises(ValueError,match="Complete binary"):global_fit(out,badx)
    with pytest.raises(ValueError,match="cohort"):global_fit(out,x.iloc[1:])

def test_cache_data_coeff_seed_and_proof_invalidation_without_refit(monkeypatch):
    import ui.irt_python_library_view as view
    import core.irt_global_fit as G
    state={};monkeypatch.setattr(view.st,"session_state",state)
    x=pd.DataFrame([[0,1,1],[1,0,1],[0,0,1]])
    co=pd.DataFrame({"discrimination":[1.]*3,"difficulty":[0.]*3})
    out=SimpleNamespace(fit=SimpleNamespace(coef=lambda:co),solver_settings={"n_quadpts":61},estimation="MAP")
    saved={"key":("original-fit",),"outcome":out};calls=[];proof=["v1"]
    def calculate(*a,**kw):
        calls.append(kw["seed"][:])
        return {"table":pd.DataFrame([{"M2":1.}]),"residuals":pd.DataFrame(np.zeros((3,3)))}
    monkeypatch.setattr(G,"global_fit",calculate);monkeypatch.setattr(G,"cache_fingerprint",lambda:proof[0])
    a=view._globalfit(saved,x,"4PL");assert view._globalfit(saved,x,"4PL") is a and len(calls)==1
    x.iloc[0,0]=1;assert view._globalfit(saved,x,"4PL") is not a and len(calls)==2
    co.iloc[0,1]=.1;view._globalfit(saved,x,"4PL");assert len(calls)==3
    saved["global_reference_seed"]=[7,8];view._globalfit(saved,x,"4PL");assert len(calls)==4
    proof[0]="withdrawn";view._globalfit(saved,x,"4PL");assert len(calls)==5
    saved["global_calibration_profile"]={"selection":"multiple starts"};view._globalfit(saved,x,"4PL");assert len(calls)==6

def ctx(family="Rasch",q=12,ll=-100.,objective="ML",**kwargs):
    return dict(model=family,family=family,N=3000,J=12,objective=objective,ll=ll,q=q,converged=True,
                boundary=False,regular=True,settings={"n_quadpts":61},latent="N(0,1) fixed",data_sha256="same",**kwargs)

def test_nested_regular_ml_and_non_wilks_cases():
    from scipy.stats import chi2
    a=ctx();b=ctx("2PL",24,-90.)
    r=nested_lrt(a,b)
    assert r["LR"]==20 and r["sd"]==12
    assert r["p"]==pytest.approx(chi2.sf(20,12)) and "NOT MC CERTIFIED" in r["Durum"]
    for change in (dict(objective="MAP"),dict(boundary=True),dict(regular=False),dict(data_sha256="other"),dict(ll=-101.)):
        z={**b,**change};assert np.isnan(nested_lrt(a,z)["p"])
    assert np.isnan(nested_lrt(ctx("3PL",36),ctx("4PL",48,-90))["p"])
    assert np.isnan(nested_lrt(a,{**b,"settings":{"n_quadpts":121}})["p"])

def test_bundle_contains_global_residuals_and_separately_named_score_sets(actual,monkeypatch):
    model,x,out=actual
    if model!="4PL":return
    import reports.irt_binary_report as B
    value=global_fit(out,x,seed=[4,5])
    report=build_report(out,model=model,items=x.columns,global_diagnostics=value)
    report.global_fit={**value,"comparisons":pd.DataFrame([nested_lrt(ctx("3PL",36),ctx("4PL",48,-90))])}
    # Export fixture is explicitly synthetic, not new scientific evidence.
    sets=pd.DataFrame({"Madde":[x.columns[0]],"Parametre":["c"],"Küme":["[0, .5]"],"Yöntem":["C(alpha), not Wald"],"Kapsam":["unit export fixture"]})
    persons=pd.DataFrame({"Yetenek (θ)":[0.],"SH (θ)":[1.]})
    monkeypatch.setattr(B,"png_bytes",lambda *a,**k:b"PNG export unit fixture")
    excel,archive=B.bundle(report,persons,None,"MAP Wald withheld",score_sets=sets)
    sheets=pd.read_excel(io.BytesIO(excel),sheet_name=None)
    assert {"Global_Uyum","Global_Korelasyon_Artiklari","ML_LRT_Karsilastirmalari","4PL_CD_Skor_Guven_Kumeleri"}<=set(sheets)
    assert sheets["ML_LRT_Karsilastirmalari"]["p"].isna().all()
    assert sheets["Global_Uyum"]["M2 asimptotik p"].isna().all()
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        assert "tablolar/4PL_CD_Skor_Guven_Kumeleri.csv" in z.namelist()
        assert "grafikler/Korelasyon artıkları.png" in z.namelist()
        assert "Wald" in z.read("4PL_yontem.txt").decode()
        assert "Küresel referans p" in z.read("tablolar/Global_Uyum.csv").decode()

def test_reproducibility_ships_global_math_sources():
    from core.reproducible import _runtime_assets
    paths={str(p).replace("\\\\","/") for p in _runtime_assets([dict(module="core.irt_global_fit",function="global_fit")])}
    assert any(p.endswith("validation/binary_global_fit/kernel.py") for p in paths)
    assert any(p.endswith("validation/binary_global_fit/analysis.py") for p in paths)

def test_actual_4pl_overidentified_global_moments():
    x=pd.read_csv("tests/fixtures/irt_b11/data_irt4pl_ok.csv").iloc[:300].astype(int)
    out=fit_python_mirt(x.to_numpy(),model="4PL",estimation="MAP",n_quadpts=61)
    assert out.fit.converged
    value=global_fit(out,x,seed=[9721,4]);r=value["table"].iloc[0]
    assert r["M2 etkin sd"]>0 and np.isfinite(r["M2"])
    assert np.isnan(r["M2 asimptotik p"]) and np.isnan(r["Küresel referans p"])

def test_actual_2pl_docx_carries_global_indices_without_refit(actual,monkeypatch):
    model,x,out=actual
    if model!="2PL":return
    from docx import Document
    from reports.irt_2pl_academic_docx import export_2pl_docx
    value=global_fit(out,x)
    # The report receives the SAME already-computed diagnostics, never a new fit.
    raw=export_2pl_docx(out,list(x.columns),global_diagnostics=value)
    doc=Document(io.BytesIO(raw))
    text=" ".join(p.text for p in doc.paragraphs)+" "+" ".join(c.text for t in doc.tables for r in t.rows for c in r.cells)
    assert "M2" in text and "SRMSR" in text and "RMSEA" in text
    assert len(doc.inline_shapes)==len(x.columns)+1

def test_no_hidden_global_helper_override():
    import ast
    from pathlib import Path
    tree=ast.parse(Path("ui/irt_python_library_view.py").read_text())
    assert sum(isinstance(n,ast.FunctionDef) and n.name=="_globalfit" for n in tree.body)==1

def test_previous_scientific_certificates_and_engines_unchanged():
    import hashlib
    from pathlib import Path
    lock=json.loads(Path("validation/binary_global_fit/PROTECTED_SOURCES.json").read_text())
    for path,sha in lock.items():
        b=Path(path).read_bytes()
        assert hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()==sha,path

@pytest.mark.parametrize("model",["3PL","4PL"])
def test_confirmed_global_production_profile_and_withdrawal(model,monkeypatch):
    import time
    import core.irt_global_fit as G
    x=pd.read_csv("tests/fixtures/irt_b11/data_irt"+model.lower()+"_ok.csv").astype(int)
    assert x.shape==(3000,12)
    out=fit_python_mirt(x.to_numpy(),model=model,estimation="MAP",n_quadpts=61,max_iter=500,tol=1e-4)
    assert out.fit.converged
    profile={"start":"native default","acceleration":"none","selection":"single fit"}
    missing=G.global_fit(out,x,seed=[9881,int(model[0])])
    assert np.isnan(missing["table"].iloc[0]["Küresel referans p"]) # no implicit cold-start claim
    elapsed=[];last=None
    for r in range(5):
        start=time.perf_counter();last=G.global_fit(out,x,seed=[9882,int(model[0]),r],calibration_profile=profile)
        elapsed.append(time.perf_counter()-start)
    cell=G._proof()["models"][model];row=last["table"].iloc[0]
    assert cell["status"]=="SCOPED CERTIFIED" and np.isfinite(row["Küresel referans p"])
    assert np.isnan(row["M2 asimptotik p"]) and "SCOPED CERTIFIED" in row["Bilimsel durum"]
    assert "MAP" in str(last["table"].iloc[0]["Küresel yöntem"])
    mutated=G.global_fit(out,x,seed=[7,8],calibration_profile={**profile,"selection":"multiple starts"})
    assert np.isnan(mutated["table"].iloc[0]["Küresel referans p"])
    proof=G._proof();proof["models"][model]["gates"]["complete"]=False
    monkeypatch.setattr(G,"_proof",lambda:proof)
    refused=G.global_fit(out,x,seed=[7,8],calibration_profile=profile)
    assert np.isnan(refused["table"].iloc[0]["Küresel referans p"])
    benchmark=dict(model=model,software_calls=5,median=float(np.median(elapsed)),P95=float(np.percentile(elapsed,95)),note="Software API benchmark; confirmation reference latency separately measured at6000 datasets/model")
    from pathlib import Path
    output=Path("/tmp/global-production");output.mkdir(parents=True,exist_ok=True)
    (output/(model+".json")).write_text(json.dumps(benchmark))
    print("PRODUCTION_LATENCY_JSON "+json.dumps(benchmark))
