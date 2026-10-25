"""Harness contracts only: no R installation or simulated calibrations in CI."""
import json
import numpy as np
import pandas as pd
import pytest
from validation.r_mirt_phase1 import harness as H


def test_missing_r_records_all_cases_without_calibrating(tmp_path, monkeypatch):
    monkeypatch.setattr(H.shutil,"which",lambda _:None)
    monkeypatch.setattr(H,"python_fit",lambda *a:pytest.fail("Unnecessary Python fit"))
    monkeypatch.setattr(H.subprocess,"run",lambda *a,**k:pytest.fail("R unexpectedly executed"))
    r=H.run(tmp_path,H.MODELS)
    assert r["status"]=="PENDING" and len(r["models"])==10
    assert all(m["status"]=="SKIPPED" and m["execution"]=="PENDING" for m in r["models"])
    for m in r["models"]:
        assert len(m["data_sha256"])==64
        assert pd.read_csv(tmp_path/m["model"]/"data.csv").shape==tuple(m["shape"])
    assert json.loads((tmp_path/"report.json").read_text())["status"]=="PENDING"


def test_map_not_compared_to_unmatched_r_ml(tmp_path):
    r=H.run(tmp_path,["3PL","4PL"],estimation="MAP",rscript="must-not-run")
    assert all(m["status"]=="NOT COMPARABLE" for m in r["models"])


def test_discrepancies_require_explicit_calibrated_tolerance():
    r=H.discrepancies([0,1],[0,2])
    assert r["status"]=="SKIPPED" and r["max_absolute"]==1 and r["max_relative"]==.5
    assert H.discrepancies([1],[1],0)["status"]=="PASS"
    assert H.discrepancies([1],[2],0)["status"]=="FAIL"
    assert H.discrepancies([np.nan],[1])["status"]=="FAIL"
    assert H.discrepancies([1,2],[1])["status"]=="FAIL"


def test_key_alignment_and_missing_traces():
    p=pd.DataFrame(dict(item=["A","B"],node=[0,0],theta=[0.,0.],category=[0,0],probability=[.3,.5],information=[.4,.8]))
    r=H.compare_curves(p,p.iloc[::-1])
    assert r["probabilities"]["max_absolute"]==0 and r["test_information"]["max_absolute"]==0
    assert H.compare_curves(p,p.iloc[:1])["status"]=="FAIL"
    assert H.compare_curves(p,pd.concat([p,p]))["status"]=="FAIL"


def test_adjacent_and_nominal_parameters_have_correct_physical_scale():
    theta=np.array([0.,1.]);a=1.3;b=np.array([.5,-.4])
    logits=np.column_stack([np.zeros(2),np.cumsum(a*(theta[:,None]-b),axis=1)])
    p=np.exp(logits-logits.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
    r=H.semantic_parameters(p,theta,"GPCM")
    np.testing.assert_allclose(r["b"],b);np.testing.assert_allclose(r["a"],a)
    r=H.semantic_parameters(p,theta,"NRM")
    np.testing.assert_allclose(r["A"],[a,2*a]);np.testing.assert_allclose(r["D"],[-a*b[0],-a*b.sum()])
    assert H.semantic_parameters(p,theta,"4PL") is None  # Never approximate c/d from endpoints.


def test_parameter_coordinates_not_silently_misaligned():
    p=H.parameter_frame(["A"],[dict(a=1.,b=[.2,.4])])
    assert H.compare_parameters(p,p.iloc[::-1])["max_absolute"]==0
    assert H.compare_parameters(p,p.iloc[:1])["status"]=="NOT COMPARABLE"


def test_binary_success_probability_is_exported_as_two_categories(monkeypatch):
    from types import SimpleNamespace
    from core import irt_python_mirt_adapter as adapter
    class Model:
        def probability(self,theta): return np.full((len(theta),3),.4)
        def information(self,theta,item_idx): return np.full(len(theta),.24)
    fit=SimpleNamespace(model=Model(),converged=True,log_likelihood=-10.,aic=32.,bic=34.,
        coef=lambda:pd.DataFrame(dict(discrimination=[1.]*3,difficulty=[0.]*3)))
    monkeypatch.setattr(adapter,"fit_python_mirt",lambda *a,**kw:SimpleNamespace(fit=fit))
    curves,meta,params=H.python_fit(pd.DataFrame(np.zeros((10,3)),columns=list("ABC")),"2PL")
    assert len(curves)==81*3*2 and set(curves.category)=={0,1}
    np.testing.assert_allclose(curves.groupby(["item","node"]).probability.sum(),1.)
    assert len(params)==3 and meta["converged"]


def test_real_r_failure_is_not_misreported_as_skipped(tmp_path,monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(H.subprocess,"run",lambda *a,**k:SimpleNamespace(returncode=1,stdout="",stderr="R fit failed"))
    r=H.run(tmp_path,["2PL"],rscript="Rscript")
    assert r["models"][0]["status"]=="FAIL" and r["models"][0]["execution"]=="EXECUTED"


def test_r_without_package_remains_pending(tmp_path,monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(H.subprocess,"run",lambda *a,**k:SimpleNamespace(returncode=42,stdout="",stderr=""))
    r=H.run(tmp_path,["2PL"],rscript="Rscript")
    assert r["models"][0]["execution"]=="PENDING"


@pytest.mark.parametrize("converged", [True,False])
def test_successful_process_requires_real_output_and_never_auto_certifies(tmp_path,monkeypatch,converged):
    """Mock file protocol, NOT an R execution/validation claim."""
    from types import SimpleNamespace
    curves=pd.DataFrame(dict(item=["M1","M1"],node=[0,0],theta=[-4.,-4.],category=[0,1],
                             probability=[.8,.2],information=[.16,.16]))
    meta=dict(converged=converged,loglik=-10.,aic=22.,bic=24.,variance=1.)
    parameters=[dict(a=1.,b=0.,c=0.,d=1.)]
    def process(args,**kw):
        out=H.Path(args[-1]);curves.to_csv(out/"curves.csv",index=False)
        pd.DataFrame([dict(meta,engine="R mirt",version="mock-only",model="2PL",D=1,estimation="ML")]).to_csv(out/"metadata.csv",index=False)
        H.parameter_frame(["M1"],parameters).to_csv(out/"parameters.csv",index=False)
        return SimpleNamespace(returncode=0,stdout="",stderr="")
    monkeypatch.setattr(H.subprocess,"run",process)
    monkeypatch.setattr(H,"python_fit",lambda *a:(curves,meta,parameters))
    r=H.run(tmp_path,["2PL"],rscript="mock-only")
    m=r["models"][0]
    assert m["status"]==("SKIPPED" if converged else "FAIL")
    assert m["comparisons"]["item_parameters"]["max_absolute"]==0
    assert m["se_ci"]["status"]=="NOT COMPARABLE"


def test_r_metadata_uses_mirt_diagnostics_not_stats_generics():
    # R mirt logLik() is bare numeric: stats AIC/BIC produce length-zero values.
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / 'validation/r_mirt_phase1/reference.R').read_text()
    for field in ('logLik', 'AIC', 'BIC'):
        assert f'extract.mirt(fit,"{field}")' in source
    assert 'aic=AIC(fit)' not in source and 'bic=BIC(fit)' not in source


@pytest.mark.parametrize('model',['3PL','4PL'])
def test_matched_profile_configures_public_ml_without_production_edits(monkeypatch,model):
    from types import SimpleNamespace
    import mirt
    class Model:
        def probability(self,theta): return np.full((len(theta),3),.4)
        def information(self,theta,item_idx): return np.full(len(theta),.24)
    fit=SimpleNamespace(model=Model(),converged=True,log_likelihood=-10.,aic=32.,bic=34.,
        coef=lambda:pd.DataFrame(dict(discrimination=[1.]*3,difficulty=[0.]*3,
            guessing=[.2]*3,upper=[.95 if model=='4PL' else 1.]*3)))
    calls=[]
    def fit_once(data,**kwargs):
        calls.append(kwargs);return fit
    monkeypatch.setattr(mirt,'fit_mirt',fit_once)
    H.python_fit(pd.DataFrame(np.zeros((10,3)),columns=list('ABC')),model,matched=True)
    assert len(calls)==1
    settings=calls[0]
    assert settings['n_quadpts']==61 and settings['max_iter']==3000 and settings['tol']==1e-8
    assert settings['priors'] is None and settings['accelerate']=='none'
    assert not settings['compute_standard_errors']
    assert np.all(settings['start_values']['guessing']==.2)
    assert ('upper' in settings['start_values']) == (model=='4PL')


def test_matched_profile_rejects_unmatched_models_and_map(tmp_path):
    for models,est in [(['2PL'],'ML'),(['3PL'],'MAP')]:
        with pytest.raises(ValueError,match='ML 3PL/4PL only'):
            H.run(tmp_path,models,estimation=est,matched=True)


@pytest.mark.parametrize('outside',[False,True])
def test_matched_report_common_likelihood_and_bound_failures(tmp_path,monkeypatch,outside):
    from types import SimpleNamespace
    source=tmp_path/'fixture.csv'
    pd.DataFrame([[0,1,0],[1,0,1],[1,1,0]],columns=list('ABC')).to_csv(source,index=False)
    monkeypatch.setitem(H.SOURCES,'4PL',str(source))
    params=[dict(a=5.1 if outside else 1.,b=0.,c=.2,d=.95)]*3
    curves=pd.DataFrame(dict(item=['A','A'],node=[0,0],theta=[-4.,-4.],category=[0,1],probability=[.8,.2],information=[.16,.16]))
    meta=dict(converged=True,loglik=-10.,aic=44.,bic=40.,variance=1.)
    def process(args,**kwargs):
        assert args[-1]=='matched'
        out=H.Path(args[-2]);assert len(pd.read_csv(out/'integration.csv'))==61
        curves.to_csv(out/'curves.csv',index=False)
        H.parameter_frame(list('ABC'),params).to_csv(out/'parameters.csv',index=False)
        pd.DataFrame([dict(meta,engine='R mirt',version='mock-only',model='4PL',D=1,estimation='ML',matched_profile=True)]).to_csv(out/'metadata.csv',index=False)
        return SimpleNamespace(returncode=0,stdout='mock',stderr='')
    monkeypatch.setattr(H.subprocess,'run',process)
    monkeypatch.setattr(H,'python_fit',lambda *a,**kw:(curves,meta,params))
    r=H.run(tmp_path/'out',['4PL'],rscript='mock-R',matched=True)
    m=r['models'][0]
    assert m['comparisons']['common_grid_loglik']['max_absolute']==0
    assert (m['status']=='FAIL')==outside
    assert m['status']!='PASS'
