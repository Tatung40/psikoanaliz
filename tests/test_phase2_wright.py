"""Real Rasch EAP/item calibration, independent posterior/plot/export checks."""
from copy import copy
from dataclasses import replace
import io,json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import mirt
from scipy.special import logsumexp
from core.irt_python_mirt_adapter import fit_python_mirt
from core.irt_rasch_wright import rasch_wright_tables,wright_figure,wright_exports


@pytest.fixture(scope='module')
def calibrated():
    x=pd.read_csv('tests/fixtures/rasch/data_rasch_main.csv').astype(int)
    outcome=fit_python_mirt(x,model='Rasch')
    assert outcome.fit.converged
    ids=[f'K{i:04}' for i in range(len(x))]
    tables=rasch_wright_tables(outcome,x,item_names=x.columns,identifiers=ids)
    return x,outcome,ids,tables


def test_actual_rasch_same_scale_plot_and_exports(calibrated,record_property):
    from mirt.estimation.quadrature import GaussHermiteQuadrature
    x,outcome,ids,tables=calibrated;fit=outcome.fit
    persons,items,h=tables['persons'],tables['items'],tables['histogram']
    coef=fit.coef()
    np.testing.assert_array_equal(coef.discrimination,np.ones(x.shape[1]))
    expected=pd.Series(coef.difficulty.to_numpy(),index=x.columns).sort_values(kind='stable')
    assert items.Madde.tolist()==expected.index.tolist()
    np.testing.assert_array_equal(items.b,expected)
    assert persons.Birey.tolist()==ids
    q=GaussHermiteQuadrature(n_points=61,n_dimensions=1)
    P=fit.model.probability(q.nodes)
    lp=np.log(q.weights)[None,:]+sum(np.log(np.where(x.iloc[:,i].to_numpy()[:,None]==1,P[:,i][None,:],1-P[:,i][None,:])) for i in range(x.shape[1]))
    post=np.exp(lp-logsumexp(lp,axis=1)[:,None])
    theta=post@q.nodes[:,0];sd=np.sqrt((post*(q.nodes[:,0][None,:]-theta[:,None])**2).sum(1))
    np.testing.assert_allclose(persons['θ'],theta,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(persons['Posterior SD'],sd,rtol=1e-12,atol=1e-12)
    assert (x.sum(axis=1)==0).any() and (x.sum(axis=1)==x.shape[1]).any()
    assert np.isfinite(persons[['θ','Posterior SD']]).all().all()
    fig=wright_figure(tables);bars,points=fig.data
    np.testing.assert_array_equal(points.y,items.b);assert list(points.text)==items.Madde.tolist()
    np.testing.assert_array_equal(bars.y,h.Merkez)
    counts,_=np.histogram(theta,bins=np.r_[h.Alt,h.Üst.iloc[-1]])
    np.testing.assert_array_equal(-np.array(bars.x),counts)
    np.testing.assert_array_equal(bars.customdata,counts);assert counts.sum()==len(x)
    assert 'EAP' in bars.name and len(fig.data)==2
    lo,hi=fig.layout.yaxis.range
    assert lo<=min(theta.min(),items.b.min()) and hi>=max(theta.max(),items.b.max())
    csv,excel=wright_exports(tables)
    for key,frame in tables.items():
        pd.testing.assert_frame_equal(pd.read_csv(io.BytesIO(csv[key])),frame,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
        pd.testing.assert_frame_equal(pd.read_excel(io.BytesIO(excel),sheet_name=key),frame,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
    record_property('actual_wright',json.dumps(dict(N=len(x),items=len(items),iterations=fit.n_iterations,converged=bool(fit.converged),
        histogram_count=int(counts.sum()),theta_min=float(theta.min()),theta_max=float(theta.max()),
        max_theta_identity_error=float(abs(persons['θ']-theta).max()),item_order=items.Madde.tolist())))
    # Optional CI artifacts use the same actual plot data; no additional fit.
    import os
    destination=os.environ.get('PHASE2_EVIDENCE_DIR')
    if destination:
        target=Path(destination);target.mkdir(parents=True,exist_ok=True)
        fig.write_html(target/'rasch_wright.html',include_plotlyjs=True)
        (target/'rasch_wright.xlsx').write_bytes(excel)
        for key,blob in csv.items():(target/f'rasch_{key}.csv').write_bytes(blob)


def test_invalid_or_nonconverged_wright_is_withheld(calibrated):
    x,outcome,ids,tables=calibrated
    fit=copy(outcome.fit);fit.converged=False
    with pytest.raises(ValueError,match='Yakınsamış'):
        rasch_wright_tables(replace(outcome,fit=fit),x,item_names=x.columns)
    missing=x.astype(float);missing.iloc[0,0]=np.nan
    with pytest.raises(ValueError,match='eksiksiz'):
        rasch_wright_tables(outcome,missing,item_names=x.columns)
    with pytest.raises(ValueError,match='kimlik'):
        rasch_wright_tables(outcome,x,item_names=x.columns,identifiers=['x']*len(x))
    invalid={k:v.copy() for k,v in tables.items()};invalid['persons'].loc[0,'θ']=np.nan
    with pytest.raises(ValueError):wright_figure(invalid)


def test_rasch_ui_settings_invalidation_and_downloads(calibrated,monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    import ui.irt_python_library_view as view
    x,outcome,ids,tables=calibrated;calls=[];exports={};person_downloads={}
    def fit(data,**kwargs):
        calls.append(kwargs)
        return fit_python_mirt(data,**kwargs)
    monkeypatch.setattr(view,'fit_python_mirt',fit)
    original=st.download_button
    def capture(label,data,*args,**kwargs):
        if str(kwargs.get('key','')).startswith('rasch_wright_'):exports[kwargs.get('key')]=data
        if str(kwargs.get('key','')).startswith('python_mirt_persons_'):person_downloads[kwargs.get('key')]=data
        return original(label,data,*args,**kwargs)
    monkeypatch.setattr(st,'download_button',capture)
    data=x.copy();data.insert(0,'ID',ids);data.index=[4]*len(data)
    at=AppTest.from_file(str(Path('app.py').resolve()),default_timeout=120)
    for k,v in dict(raw_df=data,scored_df=None,item_cols=list(x.columns),id_col='ID',group_col=None,is_dichotomous=True,nav_page='MTK: Python mirt Rasch').items():at.session_state[k]=v
    at.run()
    assert not at.checkbox(key='python_mirt_solver_advanced_Rasch').value
    at.checkbox(key='python_mirt_solver_advanced_Rasch').check().run()
    prefix='python_mirt_solver_Rasch'
    at.number_input(key=prefix+'_n_quadpts').set_value(31)
    at.number_input(key=prefix+'_tol').set_value(1e-6)
    at.number_input(key=prefix+'_max_iter').set_value(700).run()
    at.button(key='python_mirt_start_Rasch').click().run()
    at.checkbox(key='python_mirt_wright_Rasch').check().run()
    assert not at.exception and len(at.get('plotly_chart'))==1 and len(exports)==4
    assert calls[-1]['max_iter']==700 and calls[-1]['tol']==1e-6 and calls[-1]['n_quadpts']==31
    saved=at.session_state['python_mirt_fit_Rasch']['outcome']
    assert saved.solver_settings==dict(max_iter=700,tol=1e-6,n_quadpts=31) and saved.fit.converged
    expected=mirt.fscores(saved.fit,x.to_numpy(),method='EAP',n_quadpts=61)
    exported=pd.read_csv(io.BytesIO(exports['rasch_wright_persons']))
    np.testing.assert_allclose(exported['θ'],np.asarray(expected.theta).reshape(-1),rtol=1e-12,atol=1e-12)
    assert exported.Birey.tolist()==ids
    at.number_input(key=prefix+'_tol').set_value(1e-5).run()
    assert not at.get('plotly_chart') and not at.get('download_button')
    at.number_input(key=prefix+'_max_iter').set_value(1).run()
    exports.clear();person_downloads.clear()
    at.button(key='python_mirt_start_Rasch').click().run()
    assert not at.exception and not at.get('plotly_chart') and not exports and not person_downloads
    assert any('yakınsamadı' in e.value for e in at.error)
