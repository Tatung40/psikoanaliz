"""One real NRM calibration reused for nominal UI, identity and export checks."""
from dataclasses import replace
from copy import copy
from pathlib import Path
import io
import numpy as np
import pandas as pd
import pytest
from core.irt_python_mirt_adapter import fit_python_mirt


@pytest.fixture(scope='module')
def nominal_fit():
    raw=pd.read_csv('validation/evidence/p5_browser/poly_nominal.csv')
    codes=raw.replace(dict(zip('ABCD',range(4)))).astype(int)
    outcome=fit_python_mirt(codes,model='NRM',n_categories=4)
    assert outcome.fit.converged and outcome.fit.n_parameters==48
    np.testing.assert_array_equal(outcome.fit.model.slopes[:,0],np.zeros(8))
    np.testing.assert_array_equal(outcome.fit.model.intercepts[:,0],np.zeros(8))
    return raw,codes,outcome


def test_nominal_ui_never_runs_ordinal_fit_or_wright_map(nominal_fit,monkeypatch):
    import mirt,streamlit as st
    from streamlit.testing.v1 import AppTest
    import ui.irt_python_library_view as view
    raw,codes,outcome=nominal_fit;calls=[];downloads={}
    def fit(x,**kwargs):
        np.testing.assert_array_equal(x,codes);assert kwargs['model']=='NRM';calls.append(1);return outcome
    monkeypatch.setattr(view,'fit_python_mirt',fit)
    monkeypatch.setattr(mirt,'itemfit',lambda *a,**k:pytest.fail('Ordinal item fit must stay disabled for NRM'))
    scoring_calls=[];native_scoring=mirt.fscores
    def scoring(*args,**kwargs):
        assert kwargs['method']=='EAP' and kwargs['n_quadpts']==61
        scoring_calls.append(1);return native_scoring(*args,**kwargs)
    monkeypatch.setattr(mirt,'fscores',scoring)
    original=st.download_button
    def capture(label,data,*args,**kwargs):
        if label.startswith('Birey yetenekleri'):downloads[label]=data
        return original(label,data,*args,**kwargs)
    monkeypatch.setattr(st,'download_button',capture)
    data=raw.copy();ids=[f'Nominal_{i:04}' for i in range(len(data))];data.insert(0,'ID',ids);data.index=[5]*len(data)
    at=AppTest.from_file(str(Path('app.py').resolve()),default_timeout=120)
    for k,v in dict(raw_df=data,scored_df=None,item_cols=list(raw.columns),id_col='ID',group_col=None,is_dichotomous=False,nav_page='MTK: Python mirt NRM').items():at.session_state[k]=v
    at.run()
    assert at.checkbox(key='python_mirt_wright_NRM').disabled and not at.checkbox(key='python_mirt_wright_NRM').value
    assert at.checkbox(key='python_mirt_output_NRM_itemfit').disabled
    at.button(key='python_mirt_start_NRM').click().run()
    assert not at.exception and len(at.dataframe)==3 and len(downloads)==2
    assert scoring_calls
    people=at.dataframe[2].value
    assert people.Birey.tolist()==ids and people['Ham puan'].isna().all() and (people.Model=='NRM').all()
    for label,blob in downloads.items():
        parsed=pd.read_excel(io.BytesIO(blob)) if label.endswith('Excel') else pd.read_csv(io.BytesIO(blob))
        assert parsed.Birey.tolist()==ids and parsed['Ham puan'].isna().all()
        np.testing.assert_allclose(parsed['Yetenek (θ)'].round(4),people['Yetenek (θ)'])
        np.testing.assert_allclose(parsed['SH (θ)'].round(4),people['SH (θ)'])
    at.checkbox(key='python_mirt_output_NRM_curves').uncheck().run()
    assert not at.exception and len(calls)==1  # PR73 outputs do not recalibrate.
    failed=copy(outcome.fit);failed.converged=False
    at.session_state['python_mirt_fit_NRM']['outcome']=replace(outcome,fit=failed)
    downloads.clear();at.run();assert not at.exception and not downloads
    at.session_state['python_mirt_fit_NRM']['outcome']=outcome
    at.text_input(key='python_mirt_nominal_labels').set_value('D,C,B,A').run()
    assert not at.exception and not at.dataframe and not at.get('download_button')
