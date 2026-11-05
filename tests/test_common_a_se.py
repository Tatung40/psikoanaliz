"""Q4 algebra, rejection and MC accounting; no simulation campaign in tests."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from core.irt_common_a_inference import common_a_basis,parameter_inference
from validation.common_a_se_study import summarize,fit_one,OUT


def fake(cov=None):
    k=3;v=np.eye(4)*.01+np.ones((4,4))*.002 if cov is None else cov
    j=np.zeros((6,4));j[:3,0]=1;j[3:,1:]=np.eye(3)
    co=pd.DataFrame({'discrimination':np.ones(k),'difficulty':[-1.,0.,1.]},index=[f'Item_{i+1}' for i in range(k)])
    labels=[f'{p}[{i}]' for p in ('discrimination','difficulty') for i in co.index]
    group=SimpleNamespace(parameter='discrimination',items=tuple(range(k)),column=None)
    return SimpleNamespace(coef=lambda:co,vcov=j@v@j.T,vcov_labels=labels,converged=True,n_parameters=4,
        refit_recipe=SimpleNamespace(options={'constraints':(group,)})),v


def test_declared_ties_recover_full_identifiable_covariance_without_pseudoinverse():
    f,v=fake();assert np.linalg.matrix_rank(f.vcov)==4<6
    got,labels=common_a_basis(f);np.testing.assert_array_equal(got,v)
    table,why=parameter_inference(f,'1PL',['A','B','C'])
    assert why is None and len(table)==4 and (table.Parametre=='a').sum()==1
    np.testing.assert_allclose(table.SH.to_numpy(),np.sqrt(np.diag(v)),atol=0,rtol=0)
    assert table.loc[table.Parametre=='a','Madde'].iloc[0]=='Ortak (tüm maddeler)'


@pytest.mark.parametrize('kind',['no_metadata','nonconverged','singular','nan','inconsistent_copy','boundary','outside','wrong_count'])
def test_expected_embedding_nullspace_does_not_release_genuinely_invalid_inference(kind):
    f,v=fake()
    if kind=='no_metadata':f.refit_recipe=None
    elif kind=='nonconverged':f.converged=False
    elif kind=='singular':
        v[-1]=v[-2];v[:,-1]=v[:,-2];f,_=fake(v)
    elif kind=='nan':f.vcov[-1,-1]=np.nan
    elif kind=='inconsistent_copy':f.vcov[1,2]+=.001
    elif kind in ('boundary','outside'):
        co=f.coef();co['discrimination']=.1 if kind=='boundary' else .05
    elif kind=='wrong_count':f.n_parameters=6
    result,why=parameter_inference(f,'1PL',['A','B','C'])
    assert result is None and why


def test_uncertified_or_out_of_scope_common_a_stays_closed_in_production():
    f,_=fake()
    assert parameter_inference(f,'1PL',['A','B','C'],'ML')[0] is None
    from core.irt_binary_inference import parameter_inference as old
    result,why=parameter_inference(f,'2PL',['A','B','C'],'ML')
    original,original_why=old(f,'2PL',['A','B','C'],'ML')
    assert result is original is None and why==original_why


def test_shared_a_is_one_interval_and_pooled_b_uses_replication_clusters():
    p=json.loads((OUT/'protocol.json').read_text());p['R']=4
    truth=np.r_[1.,p['DGP']['b']]
    rows=[dict(index=i,converged=True,status='OK',estimate=(truth+(0 if i%2==0 else .4)).tolist(),
               se=[.1]*13,reason='') for i in range(4)]
    d=summarize(rows,p)
    assert d['parameters']['a_common']['eligible']==4
    assert d['pooled_b']['eligible']==48
    assert d['pooled_b']['coverage']==.5
    assert d['pooled_b']['coverage_mcse']==pytest.approx(np.sqrt(.25/3))
    assert d['pooled_b']['coverage_mcse']>np.sqrt(.25/48)
    assert not d['checks']['a_common_precision'] and d['status']=='NOT CERTIFIED'
    p['R']=5;rows.append(dict(index=4,converged=False,status='ERROR',estimate=[None]*13,se=[None]*13,reason='failure'))
    d=summarize(rows,p)
    assert d['convergence_rate']==.8 and d['parameters']['a_common']['missing_outcome_bounds']==[.4,.6]


def test_interrupted_claim_is_not_fitted_or_replaced(tmp_path):
    (tmp_path/'00000.claim').write_text('0')
    assert fit_one((0,str(tmp_path)))==0
    data=(tmp_path/'00000.json').read_bytes()
    assert json.loads(data)['status']=='INTERRUPTED'
    assert fit_one((0,str(tmp_path)))==0 and (tmp_path/'00000.json').read_bytes()==data


def test_preflight_independent_scale_and_full_covariance_transform():
    d=json.loads((OUT/'preflight.json').read_text())
    assert d['python_converged'] and d['native_converged'] and d['rows']==11
    assert d['max_parameter_difference']<=1e-3 and abs(d['loglik_difference'])<=1e-3
    assert d['max_relative_SE_difference']<=.02
    p=json.loads((OUT/'protocol.json').read_text())
    assert p['initial_planned_R']==300 and p['R']==4300
    assert p['precision_planning']['R300_shared_a_expected_MCSE']>.004
    assert p['R']>=p['precision_planning']['R_min_with_95pct_eligible']
