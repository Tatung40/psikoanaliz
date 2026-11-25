"""Conditional scoring validation only. No parameter-coverage certification."""
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.special import expit

from core.irt_binary_persons import binary_person_abilities

ROOT=Path(__file__).resolve().parents[1]


def moments(a,b,c,d,response,finite=False):
    response=np.asarray(response,float)
    def lp(theta):
        p=np.clip(c+(d-c)*expit(a*(theta-b)),1e-15,1-1e-15)
        return float((response*np.log(p)+(1-response)*np.log1p(-p)).sum()-.5*theta**2)
    offset=max(lp(t) for t in np.linspace(-9,9,361))
    lo,hi=(-12,12) if finite else (-np.inf,np.inf)
    values=[quad(lambda t: np.exp(lp(t)-offset)*t**k,lo,hi,epsabs=1e-10,epsrel=1e-10,limit=200)[0] for k in (0,1,2)]
    mean=values[1]/values[0]
    return np.array([mean,np.sqrt(values[2]/values[0]-mean**2)])


@pytest.mark.parametrize("model",["3PL","4PL"])
def test_native_scoring_matches_independent_integrals_and_exports(model,tmp_path,monkeypatch):
    pytest.importorskip("mirt")
    from core.irt_python_mirt_adapter import fit_python_mirt
    from core.irt_python_mirt_report import build_report
    from core.irt_library_diagnostics import library_itemfit
    from reports import irt_binary_report as reporting
    # One fit per engine model. Fixed deterministic seed; no repeated attempts.
    rng=np.random.default_rng(202611251 if model=="3PL" else 202611252)
    n,k=300,6
    a=np.linspace(.8,1.5,k);b=np.linspace(-1,1,k);c=np.full(k,.15);d=np.full(k,1 if model=="3PL" else .9)
    theta=rng.normal(size=n)
    p=c+(d-c)*expit(a*(theta[:,None]-b))
    x=(rng.random(p.shape)<p).astype(int)
    outcome=fit_python_mirt(x,model=model,estimation="MAP",n_quadpts=41,max_iter=1500,tol=1e-5)
    assert outcome.fit.converged
    patterns=np.array([[0]*k,[1]*k,[0,1]*3],int)
    ids=["sıfır","tam","karışık"]
    people=binary_person_abilities(outcome,patterns,ids)
    assert people.Birey.tolist()==ids
    co=outcome.fit.coef()
    a,b,c=[co[q].to_numpy(float) for q in ("discrimination","difficulty","guessing")]
    d=co["upper"].to_numpy(float) if "upper" in co else np.ones(k)
    diffs=[]
    for i,response in enumerate(patterns):
        ref=moments(a,b,c,d,response)
        np.testing.assert_allclose(ref,moments(a,b,c,d,response,finite=True),rtol=0,atol=1e-8)
        measured=people.iloc[i][["Yetenek (θ)","SH (θ)"]].to_numpy(float)
        np.testing.assert_allclose(measured,ref,rtol=0,atol=1e-6)
        diffs.append(float(abs(measured-ref).max()))
    assert people["Bilimsel durum"].str.contains("NOT CERTIFIED").all()
    report=build_report(outcome,model=model,items=[f"M{j+1}" for j in range(k)])
    assert report.metadata["Klasik ML model seçimi"]=="KAPALI (MAP)"
    # Unchanged S-X2 gate must not leak inference to the shared exporter.
    monkeypatch.setattr(__import__("mirt"),"itemfit",lambda *args,**kwargs:pd.DataFrame({"S_X2":[2.]*k,"p_value":[.001]*k,"p_value_adjusted":[.002]*k}))
    itemfit=library_itemfit(outcome.fit,x,model=model)
    assert itemfit.p_value.isna().all() and itemfit.p_value_adjusted.isna().all()
    # Figures are unchanged; this test exercises real numeric table/ZIP serialization.
    monkeypatch.setattr(reporting,"png_bytes",lambda *args,**kwargs:b"\x89PNG\r\n\x1a\n")
    excel,bundle=reporting.bundle(report,people,None,"SE/CI NOT CERTIFIED",itemfit)
    sheets=pd.read_excel(io.BytesIO(excel),sheet_name=None)
    assert sheets["Kisi_Yetenekleri"].Birey.tolist()==ids
    assert sheets["Madde_Uyumu"].p_value.isna().all()
    assert "MAP" in sheets["Model_Ozeti"]["Uyum ölçütü yorumu"].iloc[0]
    with zipfile.ZipFile(io.BytesIO(bundle)) as z:
        method=z.read(f"{model}_yontem.txt").decode()
        assert "klasik ML" in method and "kapsaması NOT CERTIFIED" in method
    evidence={"model":model,"fit_converged":True,"native_engine":"Python mirt 1.2.0 fscores",
              "max_difference":max(diffs),"conditional_scores":people.to_dict("records"),
              "scope":"Conditional posterior numerical moments only; NOT recovery/coverage certification",
              "CERTIFIED":False}
    (tmp_path/"comparison.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
    print("CONDITIONAL_NUMERICAL_EVIDENCE",json.dumps({"model":model,"max_difference":max(diffs),"CERTIFIED":False}))


def stub():
    return SimpleNamespace(model="3PL",n_items=3,fit=SimpleNamespace(converged=True,latent_mean=[0.],latent_covariance=[[1.]],model=SimpleNamespace(n_factors=1)))


@pytest.mark.parametrize("bad",["convergence","scale","ids","nan","unstable"])
def test_closed_scoring_gates(bad,monkeypatch):
    mirt=pytest.importorskip("mirt")
    out=stub();ids=["x"];x=np.zeros((1,3))
    if bad=="convergence":out.fit.converged=False
    if bad=="scale":out.fit.latent_covariance=[[2.]]
    if bad=="ids":ids=[]
    def score(*args,**kwargs):
        shift=.01 if bad=="unstable" and kwargs["n_quadpts"]==241 else 0.
        return SimpleNamespace(theta=np.array([np.nan if bad=="nan" else shift]),
                               standard_error=np.array([1.]),person_ids=["x"])
    monkeypatch.setattr(mirt,"fscores",score)
    with pytest.raises(ValueError):binary_person_abilities(out,x,ids)


@pytest.mark.parametrize("model,upper",[("3PL",1.),("4PL",.9)])
def test_old_finite_grid_tail_counterexample_is_preserved(model,upper):
    from core.irt_person_abilities import person_abilities
    k=12;a=np.full(k,1.4);b=np.full(k,2.5);c=np.full(k,.15);d=np.full(k,upper)
    grid=np.linspace(-4,4,161)
    rows=[]
    for j in range(k):
        p=c[j]+(d[j]-c[j])*expit(a[j]*(grid-b[j]))
        for t,v in zip(grid,p):
            rows.extend([{"Madde":f"M{j}","Theta":t,"Kategori":"0","Olasılık":1-v},
                         {"Madde":f"M{j}","Theta":t,"Kategori":"1","Olasılık":v}])
    old=person_abilities(np.ones((1,k)),pd.DataFrame(rows),model=model)
    reference=moments(a,b,c,d,np.ones(k))
    assert abs(old["Yetenek (θ)"].iloc[0]-reference[0])>.01
    assert abs(old["SH (θ)"].iloc[0]-reference[1])>.01


def test_prior_certification_registry_and_interval_policy_unchanged():
    from core.irt_binary_inference import EVIDENCE_34
    cert=json.loads((ROOT/"validation/evidence/binary_certification/CERTIFICATION.json").read_text())
    assert cert["models"]["1PL"]["status"]=="CERTIFIED"
    assert cert["models"]["2PL"]["status"]=="CERTIFIED"
    assert cert["models"]["3PL"]["status"]=="OUTPUT-LEVEL CERTIFIED"
    assert cert["models"]["4PL"]["status"]=="CERTIFIED"
    for p in ("b","c"):assert EVIDENCE_34[("3PL","MAP",p)][2] is False
    for p in ("a","b","c","d"):assert EVIDENCE_34[("4PL","MAP",p)][2] is False


def test_both_production_person_export_calls_use_public_scoring_path():
    source=(ROOT/"ui/irt_python_library_view.py").read_text()
    assert source.count("binary_person_abilities(saved['outcome'],x.to_numpy(),identifiers=ids)")==2
    assert "bundle_key=(saved['key'],SCORING_VERSION)" in source


def test_existing_registry_bytes_are_not_rewritten():
    import hashlib
    content=(ROOT/"validation/evidence/binary_certification/CERTIFICATION.json").read_bytes()
    assert hashlib.sha1(b"blob "+str(len(content)).encode()+b"\0"+content).hexdigest()=="49b5aa914c8ea99ec319ded58bf09ac6b85a4de2"
