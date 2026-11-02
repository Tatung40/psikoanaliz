"""Full covariance/link-Jacobian identities; real calibration fixtures, no MC."""
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2
from core.irt_dif import Design,_contrast_cov,_checked_covariance,wald_dif
from core.dif_lord import link_contrast,lord_from_parameters,physical_fit,lord_dif


@pytest.mark.parametrize('model',['1PL','2PL'])
def test_link_jacobians_include_anchor_uncertainty_and_match_finite_differences(model):
    r=np.array([.8,1.2,1.5,1.,-1.,0.,.8,1.7]);f=np.array([1.1,.9,1.3,1.2,-.6,.4,1.5,2.])
    if model=='1PL':r,f=r[4:],f[4:]
    delta,jr,jf,A,B=link_contrast(r,f,[0,1],2,model)
    for source,J in [(0,jr),(1,jf)]:
        numeric=np.empty_like(J)
        for i in range(len(r)):
            h=1e-6;up=[r.copy(),f.copy()];dn=[r.copy(),f.copy()]
            up[source][i]+=h;dn[source][i]-=h
            numeric[:,i]=(link_contrast(*up,[0,1],2,model)[0]-link_contrast(*dn,[0,1],2,model)[0])/(2*h)
        np.testing.assert_allclose(J,numeric,atol=1e-8,rtol=1e-7)
    # Nonzero anchor derivatives and covariance cross-terms cannot be dropped.
    assert np.linalg.norm(jr[:,(np.array([0,1])+(0 if model=='1PL' else 4))])>0
    if model=='1PL':assert A==1
    else:assert A==pytest.approx(np.std(r[4:6],ddof=1)/np.std(f[4:6],ddof=1))


@pytest.mark.parametrize('model',['1PL','2PL'])
def test_lord_exact_affine_scale_transformation_has_zero_contrast(model):
    a=np.array([.8,1.,1.4,1.2]);b=np.array([-1.,0.,1.,2.]);A=1. if model=='1PL' else 1.3;B=.4
    r=b if model=='1PL' else np.r_[a,b];f=(b-B)/A if model=='1PL' else np.r_[a*A,(b-B)/A]
    delta,*_=link_contrast(r,f,[0,1],2,model)
    np.testing.assert_allclose(delta,0,atol=1e-15)
    n=len(r);V=np.eye(n)*.05
    result=lord_from_parameters(r,f,V,V,anchors=['A','B'],studied=['C'],items=list('ABCD'),model=model)
    row=result['table'].iloc[0]
    assert row.Statistic<1e-25 and row.p==pytest.approx(1.)


def joint_fit():
    d=Design('2PL',['anchor','anchor','ab'],2)
    p=np.zeros(d.size)
    for i,(kind,_,_) in enumerate(d.labels):p[i]=1. if kind=='a' else .2*i
    v=np.linspace(.1,1.,d.size);V=np.outer(v,v)+.2*np.eye(d.size)
    return dict(design=d,params=p,cov=V,converged=True,at_bound=np.zeros(d.size,bool))


def test_wald_full_contrast_contains_cross_group_and_a_b_covariance():
    f=joint_fit();d=f['design'];i=2
    C=np.zeros((2,d.size));C[0,d.a[1,i]]=1;C[0,d.a[0,i]]=-1;C[1,d.b[1,i]]=1;C[1,d.b[0,i]]=-1
    delta,V=_contrast_cov(f,i)
    np.testing.assert_array_equal(delta,C@f['params'])
    np.testing.assert_allclose(V,C@f['cov']@C.T,atol=0,rtol=0)
    assert V[0,1]!=0
    assert V[0,0]==pytest.approx(f['cov'][d.a[0,i],d.a[0,i]]+f['cov'][d.a[1,i],d.a[1,i]]-2*f['cov'][d.a[0,i],d.a[1,i]])


@pytest.mark.parametrize('failure',['nonconverged','boundary','singular','indefinite','ill_conditioned'])
def test_wald_invalid_fits_reject_inference(failure):
    f=joint_fit()
    if failure=='nonconverged':f['converged']=False
    elif failure=='boundary':f['at_bound'][0]=True
    elif failure=='singular':f['cov'][:]=0
    elif failure=='indefinite':f['cov'][0,0]=-1
    else:f['cov']=np.diag([1e-10]+[1.]*(f['design'].size-1))
    with pytest.raises(ValueError):_contrast_cov(f,2)


def test_lord_singular_anchor_spread_and_bad_covariance_return_na():
    items=list('ABCD');r=np.r_[np.ones(4),[-1.,0.,1.,2.]];f=r.copy();V=np.eye(8)
    result=lord_from_parameters(r,f,V*0,V,anchors=['A','B'],studied=['C'],items=items,model='2PL')
    assert result['table'].p.isna().all() and result['table'].Statistic.isna().all()
    f[4:6]=0
    result=lord_from_parameters(r,f,V,V,anchors=['A','B'],studied=['C'],items=items,model='2PL')
    assert result['table'].p.isna().all() and 'yayılım' in result['table'].Reason.iloc[0]


@pytest.mark.parametrize('method',[wald_dif,lord_dif])
def test_unsupported_3pl_rejected_before_calibration(method):
    with pytest.raises(ValueError,match='1PL/2PL'):
        method(pd.DataFrame(np.ones((60,4)),columns=list('ABCD')),['R']*30+['F']*30,'R','F',
               model='3PL',anchors=['A','B'],studied=['C'])


@pytest.mark.parametrize('model',['1PL','2PL'])
def test_physical_covariance_label_reordering_and_fixed_parameters(model):
    names=['I1','I2','I3'];a=np.ones(3);b=np.array([-1.,0.,1.])
    wanted=([f'discrimination[{n}]' for n in names] if model=='2PL' else [])+[f'difficulty[{n}]' for n in names]
    labels=wanted[::-1];n=len(labels);cov=np.diag(np.arange(1,n+1)/10)
    fit=SimpleNamespace(converged=True,coef=lambda:pd.DataFrame({'discrimination':a,'difficulty':b},index=names),
                        vcov=cov,vcov_labels=labels)
    v,V=physical_fit(fit,model)
    np.testing.assert_array_equal(v,b if model=='1PL' else np.r_[a,b])
    np.testing.assert_array_equal(V,cov[::-1,::-1])
    fit.converged=False
    with pytest.raises(ValueError,match='yakınsamadı'):physical_fit(fit,model)
    fit.converged=True;b[0]=-6
    with pytest.raises(ValueError,match='Sınırda'):physical_fit(fit,model)


@pytest.mark.parametrize('case,model',[('dif1pl_uniform','1PL'),('dif2pl_mixed','2PL'),('dif2pl_none','2PL')])
def test_saved_actual_fits_reproduce_both_statistics_and_full_covariances(case,model):
    import json
    from pathlib import Path
    root=Path('validation/dif_wald_lord/results')/case
    cfg=json.loads((root/'settings.json').read_text());saved=np.load(root/'python_fits.npz')
    assert cfg['native_diagnostics']['converged'] and all(d['converged'] for d in cfg['Lord_diagnostics'])
    fit=dict(design=Design(model,['anchor' if i in cfg['anchors'] else 'b' if model=='1PL' else 'ab' for i in cfg['items']],2),
             params=saved['w_params'],cov=saved['w_cov'],at_bound=saved['w_bound'],converged=True)
    w=pd.read_csv(root/'python_wald.csv').set_index('Item')
    for item in cfg['studied']:
        delta,V=_contrast_cov(fit,cfg['items'].index(item))
        assert float(delta@np.linalg.solve(V,delta))==pytest.approx(w.loc[item,'Statistic'],rel=1e-12)
    l=lord_from_parameters(saved['r'],saved['f'],saved['Vr'],saved['Vf'],anchors=cfg['anchors'],studied=cfg['studied'],items=cfg['items'],model=model)
    np.testing.assert_allclose(l['table'].Statistic,pd.read_csv(root/'python_lord.csv').Statistic,rtol=1e-12)
    cov=l['covariance']
    np.testing.assert_allclose(cov.Covariance,cov.Reference_contribution+cov.Focal_contribution,rtol=1e-14)
    assert cfg['used_n']==(741 if model=='1PL' else 2000)


def test_center_uses_cached_actual_calibrations_and_exports_covariance(monkeypatch):
    import io,json,zipfile
    from pathlib import Path
    from core.dif_center import observed
    from reports.dif_center_report import downloads
    root=Path('validation/dif_wald_lord/results/dif2pl_mixed');cfg=json.loads((root/'settings.json').read_text())
    frame=pd.read_csv('tests/fixtures/irt_b10/data_dif2pl_mixed.csv')
    for name,fn in [('wald','core.irt_dif.wald_dif'),('lord','core.dif_lord.lord_dif')]:
        output=dict(table=pd.read_csv(root/f'python_{name}.csv',dtype={'Reason':'string'}).fillna({'Reason':''}),covariance=pd.read_csv(root/f'python_{name}_cov.csv'))
        monkeypatch.setattr(fn,lambda *a,_output=output,**kw:_output)
    result=observed(frame,items=cfg['items'],group='Grup',reference='R',focal='F',anchors=cfg['anchors'],studied=cfg['studied'],
                    methods=['Wald DIF','Lord χ²'],model='2PL')
    assert len(result['table'])==4 and result['table'].p.notna().all()
    blob=downloads(result)
    workbook=pd.read_excel(io.BytesIO(blob['xlsx']),sheet_name=None)
    assert {'Lord_covariance','Wald_covariance'}<=set(workbook)
    with zipfile.ZipFile(io.BytesIO(blob['zip'])) as z:
        assert {'Lord_covariance.csv','Wald_covariance.csv'}<=set(z.namelist())
    assert set(result['table'].Method)=={'Lord χ²','Wald DIF'}


def test_parametric_model_changes_invalidate_ui_without_refitting():
    from tests.test_dif_center_ui import app
    at=app();at.button(key='dc_run').click().run()
    assert 'dif_center_result' in at.session_state
    at.selectbox(key='dc_parametric_model').set_value('1PL').run()
    assert not at.exception and 'dif_center_result' not in at.session_state
    assert not at.get('download_button')


def test_ui_lord_wald_report_reuses_actual_saved_fits(monkeypatch):
    import json
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    root=Path('validation/dif_wald_lord/results/dif2pl_mixed');cfg=json.loads((root/'settings.json').read_text())
    data=pd.read_csv('tests/fixtures/irt_b10/data_dif2pl_mixed.csv')
    for name,target in [('wald','core.irt_dif.wald_dif'),('lord','core.dif_lord.lord_dif')]:
        result=dict(table=pd.read_csv(root/f'python_{name}.csv',dtype={'Reason':'string'}).fillna({'Reason':''}),
                    covariance=pd.read_csv(root/f'python_{name}_cov.csv'))
        def cached(frame,group,reference,focal,_result=result,**kw):
            pd.testing.assert_frame_equal(frame,data[cfg['items']])
            assert kw['anchors']==cfg['anchors'] and kw['studied']==cfg['studied'] and kw['model']=='2PL'
            return _result
        monkeypatch.setattr(target,cached)
    at=AppTest.from_file(str(Path('app.py').resolve()),default_timeout=60)
    values=dict(raw_df=data,scored_df=data,item_cols=cfg['items'],id_col=None,group_col='Grup',
        is_dichotomous=True,nav_page='MTK: DIF Analiz Merkezi',dc_methods=['Wald DIF','Lord χ²'],
        dc_anchors=cfg['anchors'],dc_studied=cfg['studied'])
    for key,value in values.items():at.session_state[key]=value
    at.run();at.button(key='dc_run').click().run()
    assert not at.exception and not at.error
    tables=[d.value for d in at.dataframe]
    table=next(f for f in tables if {'Method','p_adj','Delta_b'}<=set(f))
    assert set(table.Method)=={'Wald DIF','Lord χ²'} and table.p.notna().all()
    assert sum('Covariance' in t for t in tables)==2
    assert len(at.get('download_button'))>=5
