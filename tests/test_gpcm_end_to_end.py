"""One real GPCM fit reused for person-service and real Streamlit output checks."""
from dataclasses import replace
from copy import copy
from pathlib import Path
import io
import numpy as np
import pandas as pd
import pytest
from core.irt_python_mirt_adapter import fit_python_mirt
from core.irt_python_mirt_report import build_report
from core.irt_b14_model_curves import fitted_category_curves
from core.irt_person_abilities import person_abilities
from mirt.estimation.quadrature import GaussHermiteQuadrature

@pytest.fixture(scope='module')
def fitted():
    x=pd.read_csv('validation/evidence/p5_browser/poly_ordinal.csv')
    fit=fit_python_mirt(x,model='GPCM',n_categories=4)
    assert fit.fit.converged and fit.fit.n_parameters==32
    q=GaussHermiteQuadrature(n_points=61,n_dimensions=1)
    c=fitted_category_curves(dict(result=fit.fit,k=8,items=x.columns,labels=['0','1','2','3']),q.nodes[:,0])
    ids=[f'P{i:04}' for i in range(len(x))]
    people=person_abilities(x,c,model='GPCM',identifiers=ids,quadrature=(q.nodes[:,0],q.weights))
    return x,fit,ids,people


def test_gpcm_free_slopes_persons_and_parameter_information(fitted):
    x,fit,ids,people=fitted
    assert not np.array_equal(fit.fit.model.discrimination,np.ones(8))
    r=build_report(fit,model='GPCM',items=x.columns,labels=['0','1','2','3'])
    assert not r.errors and len(r.parameters)==8
    np.testing.assert_array_equal(r.parameters.a,fit.fit.model.discrimination)
    np.testing.assert_array_equal(r.parameters.filter(regex=' b').to_numpy(),fit.fit.model.steps)
    assert people.Birey.tolist()==ids and np.isfinite(people['Yetenek (θ)']).all()
    assert (people['SH (θ)']>0).all() and (people.Model=='GPCM').all()


def test_gpcm_person_ui_ids_exports_and_invalid_calibration(fitted,monkeypatch):
    from streamlit.testing.v1 import AppTest
    import streamlit as st
    import ui.irt_python_library_view as view
    x,fit,ids,people=fitted;calls=[];downloads={}
    def calibrate(responses,**kw):
        np.testing.assert_array_equal(responses,x)
        assert kw['model']=='GPCM';calls.append(1);return fit
    monkeypatch.setattr(view,'fit_python_mirt',calibrate)
    real_download=st.download_button
    def capture(label,data,*args,**kw):
        if label.startswith('Birey yetenekleri'):downloads[label]=data
        return real_download(label,data,*args,**kw)
    monkeypatch.setattr(st,'download_button',capture)
    data=x.copy();data.insert(0,'ID',ids);data.index=[4]*len(data)
    at=AppTest.from_file(str(Path('app.py').resolve()),default_timeout=120)
    for k,v in dict(raw_df=data,scored_df=None,item_cols=list(x.columns),id_col='ID',group_col=None,is_dichotomous=False,nav_page='MTK: Python mirt GPCM').items():at.session_state[k]=v
    at.run();at.button(key='python_mirt_start_GPCM').click().run()
    assert not at.exception and len(at.dataframe)==3
    assert at.dataframe[2].value.Birey.tolist()==ids
    for label,blob in downloads.items():
        parsed=pd.read_excel(io.BytesIO(blob)) if label.endswith('Excel') else pd.read_csv(io.BytesIO(blob))
        pd.testing.assert_frame_equal(parsed,people,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
    assert len(downloads)==2
    at.run();assert len(calls)==1
    failed=copy(fit.fit);failed.converged=False
    from core.irt_b14_persons_info import person_scores
    with pytest.raises(ValueError,match='yakınsamadı'):person_scores(dict(result=failed,coded=x))
    at.session_state['python_mirt_fit_GPCM']['outcome']=replace(fit,fit=failed)
    downloads.clear();at.run()
    assert not at.exception and len(at.dataframe)==2 and not downloads
    at.session_state['python_mirt_fit_GPCM']['outcome']=fit
    bad=data.copy();bad.iloc[0,0]=ids[1]
    at.session_state['raw_df']=bad;at.run()
    assert any('benzersiz' in e.value for e in at.error) and not downloads
