"""Cheap independent identities + real public-library APIs; no simulations."""
import io
import zipfile
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.special import expit, logsumexp
from scipy.stats import chi2
from core.dif_center import observed,raju_area,theta_sensitivity,support_matrix
from reports.dif_center_report import downloads


def fixture_data():
    frame=pd.read_csv('tests/fixtures/irt_b10/data_dif2pl_mixed.csv')
    return frame.groupby('Grup',sort=False).head(100).reset_index(drop=True)


def request(frame):
    items=[c for c in frame if c!='Grup'][:6]
    return dict(items=items,group='Grup',reference='R',focal='F',anchors=items[:4],studied=items[4:])


def test_uncorrected_sibtest_against_hand_calculated_strata():
    # Two equally weighted strata, n=30/group/stratum, p_R=2/3, p_F=1/3.
    rows=[]
    for group,pattern in [('R',[0,1,1]),('F',[0,0,1])]:
        for anchor in (0,1):
            rows.extend([dict(A=anchor,B=anchor,Y=y,G=group) for y in pattern*10])
    frame=pd.DataFrame(rows)
    result=observed(frame,items=['A','B','Y'],group='G',reference='R',focal='F',anchors=['A','B'],
                    studied=['Y'],methods=['SIBTEST'],sib_correction=False)
    row=result['table'].iloc[0]
    expected_se=np.sqrt(.5*(2*((2/9)*30/29)/30))
    assert row.Effect==pytest.approx(1/3,abs=1e-15)
    assert row.beta_se==pytest.approx(expected_se,rel=1e-14)
    assert row.Statistic==pytest.approx(14.5,rel=1e-14)
    assert row.p==pytest.approx(chi2.sf(14.5,1),rel=1e-13)
    assert row.Direction=='Referans lehine' and row.Effect_class=='UNVERIFIED'
    reverse=observed(frame,items=['A','B','Y'],group='G',reference='F',focal='R',anchors=['A','B'],
                     studied=['Y'],methods=['SIBTEST'],sib_correction=False)['table'].iloc[0]
    assert reverse.Effect==pytest.approx(-row.Effect) and reverse.p==pytest.approx(row.p)


@pytest.mark.parametrize('r,f',[
    ([1.,-.4,0.,1.],[1.,.5,0.,1.]),
    ([.8,-.5,0.,1.],[1.6,.7,0.,1.]),
    ([1.2,.3,.15,.9],[.7,-.4,.15,.9]),
    ([1.,0.,0.,1.],[1.,0.,0.,1.]),
])
def test_raju_full_line_area_against_independent_adaptive_quadrature(r,f):
    value=lambda t:(f[2]+(f[3]-f[2])*expit(f[0]*(t-f[1])))-(r[2]+(r[3]-r[2])*expit(r[0]*(t-r[1])))
    signed=quad(value,-np.inf,np.inf,epsabs=1e-11)[0]
    crossing=(f[0]*f[1]-r[0]*r[1])/(f[0]-r[0]) if f[0]!=r[0] else 0.
    unsigned=sum(quad(lambda t:abs(value(t)),a,b,epsabs=1e-11)[0] for a,b in [(-np.inf,crossing),(crossing,np.inf)])
    result=raju_area(r,f)
    assert result['Signed_area']==pytest.approx(signed,abs=1e-10,rel=1e-10)
    assert result['Unsigned_area']==pytest.approx(unsigned,abs=1e-10,rel=1e-10)
    assert np.isnan(result['p'])


def test_raju_rejects_divergent_asymptotes_and_handles_near_parallel_slopes():
    with pytest.raises(ValueError,match='ıraksar'):raju_area([1,0,0,1],[1,0,.1,1])
    result=raju_area([1.,-.5,0,1],[np.nextafter(1.,2.),.5,0,1])
    assert result['Signed_area']==-1 and result['Unsigned_area']==pytest.approx(1,abs=1e-14)


def test_real_observed_methods_adjustments_and_exports():
    from statsmodels.stats.multitest import multipletests
    frame=fixture_data();args=request(frame)
    result=observed(frame,**args)
    table=result['table']
    assert len(table)==6 and set(table.Method)=={'MH','LR','SIBTEST'}
    assert np.isfinite(table.p).all(),table[['Method','Reason']]
    for _,part in table.groupby('Method'):
        np.testing.assert_allclose(part.p_adj,multipletests(part.p,method='holm')[1],rtol=0,atol=0)
    from core.dif_classical import jodoin_gierl
    for row in table[table.Method=='LR'].itertuples():
        assert row.Effect_class==jodoin_gierl(row.Effect,row.p_adj,.05)
    payload=downloads(result)
    pd.testing.assert_frame_equal(pd.read_csv(io.BytesIO(payload['csv'])),pd.read_excel(io.BytesIO(payload['xlsx']),sheet_name='DIF'),check_dtype=False)
    with zipfile.ZipFile(io.BytesIO(payload['zip'])) as z:
        assert {'DIF.csv','report.xlsx','comparison.png','report.md'}<=set(z.namelist())
        assert 'FWER NOT CERTIFIED' in z.read('report.md').decode()
    assert set(support_matrix().Method)=={'SIBTEST','IRT-LR','Lord χ²','Raju','MH','Lojistik LR','Wald DIF'}


def test_invalid_statistics_remain_unestimable_in_every_export():
    frame=fixture_data();args=request(frame)
    import mirt
    raw=dict(beta=np.nan,beta_se=np.nan,z=np.nan,p_value=np.nan,chi2=np.nan,df=0,
             effect_size=np.nan,n_strata=0)
    with patch.object(mirt,'sibtest',return_value=raw):
        result=observed(frame,**args,methods=['SIBTEST'])
    assert result['table'][['p','p_adj']].isna().all().all()
    assert (result['table'].Decision=='Hesaplanamadı').all()
    assert (result['table'].Effect_class=='Hesaplanamadı').all()
    payload=downloads(result)
    assert pd.read_csv(io.BytesIO(payload['csv'])).p.isna().all()
    assert pd.read_excel(io.BytesIO(payload['xlsx']),sheet_name='DIF').p.isna().all()


def test_theta_sensitivity_one_real_fit_and_independent_gh61_posterior():
    from core.irt_python_mirt_adapter import fit_python_mirt
    from mirt.estimation.quadrature import GaussHermiteQuadrature
    frame=fixture_data();items=request(frame)['items'];captured=[]
    def fit(*a,**kw):
        result=fit_python_mirt(*a,**kw);captured.append(result);return result
    with patch('core.irt_python_mirt_adapter.fit_python_mirt',side_effect=fit):
        result=theta_sensitivity(frame,items=items,removed=items[-2:],model='Rasch',identifiers=['ID'+str(i) for i in frame.index])
    assert len(captured)==1
    q=GaussHermiteQuadrature(n_points=61,n_dimensions=1)
    p=captured[0].fit.model.probability(q.nodes);x=frame[items].to_numpy()
    for columns,name in [(list(range(6)),'before'),(list(range(4)),'after')]:
        lp=np.log(q.weights)[None,:]+sum(np.log(np.where(x[:,j,None]==1,p[None,:,j],1-p[None,:,j])) for j in columns)
        posterior=np.exp(lp-logsumexp(lp,axis=1)[:,None]);theta=posterior@q.nodes[:,0]
        np.testing.assert_allclose(result['Theta_'+name],theta,atol=1e-12,rtol=1e-12)
    assert result.ID.tolist()==['ID'+str(i) for i in frame.index]
    np.testing.assert_array_equal(result.Delta_theta,result.Theta_after-result.Theta_before)
    assert 'RMSD' not in result


def test_crossing_sibtest_uses_region_test_and_unsigned_beta_cross():
    rows=[]
    for group in ('R','F'):
        for anchor in (0,1):
            pattern=([0,1,1] if (group=='R')==(anchor==0) else [0,0,1])
            rows.extend([dict(A=anchor,B=anchor,Y=y,G=group) for y in pattern*10])
    result=observed(pd.DataFrame(rows),items=['A','B','Y'],group='G',reference='R',focal='F',
        anchors=['A','B'],studied=['Y'],methods=['SIBTEST'],sib_correction=False,sib_method='crossing')['table'].iloc[0]
    # Net regional effect is zero, but beta_cross=|1/6 - (-1/6)|=1/3.
    assert result.beta==pytest.approx(1/3,abs=1e-15)
    assert result.Statistic==pytest.approx(14.5,rel=1e-14) and result.df==2
    assert result.p==pytest.approx(chi2.sf(14.5,2),rel=1e-14)
    assert 'Kesişimli' in result.Direction


def test_invalid_hypothesis_does_not_shrink_planned_multiplicity_family():
    import mirt
    frame=fixture_data();args=request(frame)
    good=dict(beta=.2,beta_se=.1,z=2.,p_value=chi2.sf(4,1),chi2=4.,df=1)
    invalid=dict(beta=np.nan,beta_se=np.nan,z=np.nan,p_value=np.nan,chi2=np.nan,df=0)
    with patch.object(mirt,'sibtest',side_effect=[good,invalid]):
        result=observed(frame,**args,methods=['SIBTEST'])['table']
    assert result.p_adj.iloc[0]==pytest.approx(2*good['p_value'])
    assert np.isnan(result.p_adj.iloc[1])
