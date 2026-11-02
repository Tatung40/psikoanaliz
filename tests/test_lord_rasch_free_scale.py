"""No reference fabrication or repeated fits: algebra, gates, saved-result policy."""
import numpy as np
import pandas as pd
import pytest
from core import dif_lord_rasch as L
from core.dif_lord_policy import protect_result
from core.dif_lord import link_contrast


def mocked_fit(monkeypatch):
    fit=dict(converged=True,at_bound=False,se_reason='',b=np.array([-.5,.2,.7]),sd=1.4,
        data=np.tile([0.,1.,0.],(50,1)),points=61,loglik=-40.,max_score_per_person=1e-8)
    info=np.eye(4);info[0,3]=info[3,0]=.5
    monkeypatch.setattr(L.rasch,'_information',lambda *a:info)
    monkeypatch.setattr(L.rasch,'_score',lambda *a:np.zeros(4))
    monkeypatch.setattr(L.rasch,'marginal_loglik',lambda *a:-40.)
    return fit,info


def test_full_inverse_nuisance_block_is_not_conditional_information(monkeypatch):
    fit,info=mocked_fit(monkeypatch);b,bb,full,diag=L.rasch_physical_fit(fit)
    np.testing.assert_allclose(full,np.linalg.inv(info));np.testing.assert_allclose(bb,full[:3,:3])
    assert bb[0,0]>np.linalg.inv(info[:3,:3])[0,0]
    assert diag['sd']==1.4 and diag['GH101_score']==0
    # A=1 depends on fixed slopes, not equality of latent SDs.
    d,jr,jf,A,B=link_contrast(b,b+.3,[0,1],2,'1PL')
    assert A==1;np.testing.assert_allclose(d,0,atol=1e-15)


@pytest.mark.parametrize('failure',['nonconverged','bound','singular','illconditioned','audit'])
def test_unchanged_scientific_covariance_and_score_gates(monkeypatch,failure):
    fit,info=mocked_fit(monkeypatch)
    if failure=='nonconverged':fit['converged']=False
    if failure=='bound':fit['at_bound']=True
    if failure=='singular':monkeypatch.setattr(L.rasch,'_information',lambda *a:np.zeros((4,4)))
    if failure=='illconditioned':monkeypatch.setattr(L.rasch,'_information',lambda *a:np.diag([1,1,1,1e-10]))
    if failure=='audit':monkeypatch.setattr(L.rasch,'_score',lambda *a:np.ones(4))
    with pytest.raises(ValueError):L.rasch_physical_fit(fit)


def test_free_scale_inference_requires_actual_independent_reference(monkeypatch):
    result=dict(table=pd.DataFrame([dict(Method='Lord χ²',p=.02,q=.04,p_adj=.04,df=1,Statistic=5.,
        Lord_identification=L.IDENTIFICATION,Status=L.STATUS,Reason='')]),metadata={'Parametric_model':'1PL'})
    monkeypatch.setattr(L,'reference_valid',lambda:False)
    assert protect_result(result)['table'].p.isna().all()
    monkeypatch.setattr(L,'reference_valid',lambda:True)
    assert protect_result(result) is result
    legacy={**result,'table':result['table'].drop(columns='Lord_identification')}
    assert protect_result(legacy)['table'].p.isna().all()


def test_no_reference_cannot_be_assumed_from_model_name(monkeypatch,tmp_path):
    monkeypatch.setattr(L,'REFERENCE',tmp_path/'missing.json')
    assert L.reference_valid() is False


def test_unestimable_rows_keep_failure_status():
    raw=dict(table=pd.DataFrame([dict(Method='Lord χ²',Statistic=np.nan,p=np.nan,
        Status='UNESTIMABLE / NOT CERTIFIED',Reason='Rasch yakınsama kapısı')]),metadata={'Parametric_model':'1PL'})
    result=protect_result(raw)
    assert result['table'].Status.eq('UNESTIMABLE / NOT CERTIFIED').all()
    assert result['table'].p.isna().all()


def test_actual_independent_free_sd_evidence_and_marginal_covariance():
    import json
    root=L.REFERENCE.parent
    assert L.reference_valid()
    report=json.loads(L.REFERENCE.read_text())
    for q in ('61','101'):
        result=report['comparisons'][q]
        assert result['Statistic']['max_relative']<=1e-3
        for g in ('R','F'):
            assert result[g]['R_solution_common_GH101_score']<=1e-5
            folder=root/f'GH{q}'
            full=pd.read_csv(folder/f'python_{g}_full_cov.csv').to_numpy()
            bb=pd.read_csv(folder/f'python_{g}_b_cov.csv').to_numpy()
            np.testing.assert_array_equal(full[:-1,:-1],bb)
            conditional=np.linalg.inv(np.linalg.inv(full)[:-1,:-1])
            assert np.max(np.abs(bb-conditional))>0
            values=pd.read_csv(root/'wide'/f'GH{q}'/f'r_{g}_raw_parameters.csv')
            assert values.loc[values.name.eq('COV_11'),'est'].item()
            assert not values.loc[values.name.eq('MEAN_1'),'est'].item()
            assert values.loc[values.name.eq('a1'),'value'].eq(1).all()


@pytest.mark.parametrize('failure',['window','score','chi','convergence'])
def test_pass_label_cannot_override_failed_reference_criteria(monkeypatch,tmp_path,failure):
    import json
    from core.dif_lord_policy import free_scale_reference_eligible
    report=json.loads(L.REFERENCE.read_text())
    if failure=='window':report['R_window']='default[-6,6]'
    if failure=='score':report['comparisons']['61']['F']['R_solution_common_GH101_score']=1.01e-5
    if failure=='chi':report['comparisons']['101']['Statistic']['max_relative']=1.01e-3
    if failure=='convergence':report['comparisons']['61']['R']['R_diagnostics']['converged']=False
    fake=tmp_path/'report.json';fake.write_text(json.dumps(report));monkeypatch.setattr(L,'REFERENCE',fake)
    monkeypatch.setattr(L,'reference_valid',lambda:True)
    assert not free_scale_reference_eligible()


def test_real_free_sd_center_and_exports_allow_only_the_new_valid_path(monkeypatch):
    import io,json,zipfile
    from core.dif_center import observed
    from reports.dif_center_report import downloads
    folder=L.REFERENCE.parent/'GH61'
    raw=pd.read_csv(folder/'python_lord.csv').fillna({'Reason':''})
    outcome=dict(table=raw,covariance=pd.read_csv(folder/'python_lord_cov.csv'),
        diagnostics=json.loads((folder/'python_diagnostics.json').read_text()))
    monkeypatch.setattr(L,'lord_1pl_rasch',lambda *a,**kw:outcome)
    settings=json.loads((L.REFERENCE.parent/'settings.json').read_text())
    data=pd.read_csv('tests/fixtures/irt_b10/data_dif1pl_uniform.csv').dropna()
    result=observed(data,items=settings['items'],group='Grup',reference='R',focal='F',
        anchors=settings['anchors'],studied=settings['studied'],methods=['Lord χ²'],model='1PL',correction='BH')
    assert result['table'].p.notna().all() and result['table'].p_adj.notna().all()
    assert result['table'].Status.eq(L.STATUS).all()
    assert result['table'].Lord_sigma_reference.eq(outcome['diagnostics'][0]['sd']).all()
    assert result['table'].Lord_sigma_focal.eq(outcome['diagnostics'][1]['sd']).all()
    assert len(result['extra_tables']['Lord_group_diagnostics'])==2
    payload=downloads(result)
    tables=[pd.read_csv(io.BytesIO(payload['csv'])),pd.read_excel(io.BytesIO(payload['xlsx']),sheet_name='DIF')]
    with zipfile.ZipFile(io.BytesIO(payload['zip'])) as z:tables.append(pd.read_csv(io.BytesIO(z.read('DIF.csv'))))
    for table in tables:
        np.testing.assert_allclose(table.p,result['table'].p,rtol=1e-13)
        assert table.Status.eq(L.STATUS).all()
    # Existing legacy-table tests separately prove that reference PASS cannot unlock fixed-SD outputs.
