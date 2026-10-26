"""Real fitted inputs: SPSS controls preserve numerical values and never refit."""
from pathlib import Path
import io
import numpy as np
import pandas as pd
import pytest
from core.irt_python_mirt_adapter import fit_python_mirt

@pytest.fixture(scope='module')
def actual_fits():
    x=pd.read_csv('tests/fixtures/irt_b10/data_irt1pl_ok.csv').iloc[:240,:6].astype(int)
    fits={(model,q):fit_python_mirt(x,model=model,n_quadpts=q)
          for model,q in [('Rasch',21),('2PL',21),('2PL',31)]}
    assert all(v.fit.converged for v in fits.values())
    return x,fits


def app_with_real_fits(actual_fits,monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    import ui.irt_python_library_view as view
    x,fits=actual_fits;calls=[];downloads={}
    def calibrate(data,**kwargs):
        np.testing.assert_array_equal(data,x.to_numpy())
        calls.append(kwargs)
        if kwargs['tol']==0:return fit_python_mirt(data,**kwargs)  # actual fail-closed validation, no fit.
        assert kwargs['max_iter']==500 and kwargs['tol']==1e-4
        return fits[kwargs['model'],kwargs['n_quadpts']]
    monkeypatch.setattr(view,'fit_python_mirt',calibrate)
    original=st.download_button
    def capture(label,data,*args,**kwargs):
        downloads[kwargs.get('key')]=data
        return original(label,data,*args,**kwargs)
    monkeypatch.setattr(st,'download_button',capture)
    data=x.copy();data.insert(0,'ID',[f'P{i:04}' for i in range(len(x))]);data.index=[4]*len(data)
    at=AppTest.from_file(str(Path('app.py').resolve()),default_timeout=120)
    for k,v in dict(raw_df=data,scored_df=None,item_cols=list(x.columns),id_col='ID',group_col=None,is_dichotomous=True,nav_page='MTK: Python mirt Rasch').items():at.session_state[k]=v
    at.run();assert not at.exception
    assert all(f'python_irt_{m}' in {b.key for b in at.button} for m in ('Rasch','1PL','2PL','3PL','4PL','GRM','PCM','GPCM','NRM'))
    return at,calls,downloads,data


def test_spss_outputs_and_rasch_map_preserve_actual_values(actual_fits,monkeypatch):
    at,calls,downloads,data=app_with_real_fits(actual_fits,monkeypatch)
    outputs=['summary','parameters','persons','curves','information','comparison','itemfit']
    assert all(at.checkbox(key=f'python_mirt_output_Rasch_{k}').value for k in outputs)
    assert not at.checkbox(key='python_mirt_solver_advanced_Rasch').value
    at.button(key='python_mirt_start_Rasch').click().run()
    assert not at.exception and len(calls)==1 and len(at.dataframe)==4
    params=downloads['python_mirt_parameters_csv_Rasch'];people=downloads['python_mirt_persons_csv_Rasch']
    for group,key in [('summary','summary'),('parameters','parameters'),('persons','persons'),('curves','curves'),('information','information'),('itemfit','itemfit')]:
        downloads.clear();at.checkbox(key=f'python_mirt_output_Rasch_{group}').uncheck().run()
        assert not at.exception and f'python_mirt_{key}_csv_Rasch' not in downloads and len(calls)==1
        at.checkbox(key=f'python_mirt_output_Rasch_{group}').check().run()
        assert not at.exception and len(calls)==1
        assert downloads['python_mirt_parameters_csv_Rasch']==params
        assert downloads['python_mirt_persons_csv_Rasch']==people
    at.checkbox(key='python_mirt_wright_Rasch').check().run()
    assert not at.exception and len(at.get('plotly_chart'))==1 and len(calls)==1
    native=pd.read_csv(io.BytesIO(downloads['python_mirt_persons_csv_Rasch']))
    wright=pd.read_csv(io.BytesIO(downloads['rasch_wright_persons']))
    np.testing.assert_array_equal(native.Birey,wright.Birey)
    np.testing.assert_allclose(native['Yetenek (θ)'],wright['θ'],rtol=0,atol=1e-14)
    np.testing.assert_allclose(native['SH (θ)'],wright['Posterior SD'],rtol=0,atol=1e-14)
    excel=pd.read_excel(io.BytesIO(downloads['rasch_wright_excel']),sheet_name='persons')
    pd.testing.assert_frame_equal(excel,wright,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
    downloads.clear()
    changed=data.copy();changed.iloc[0,0]='Changed_ID'
    at.session_state['raw_df']=changed;at.run()
    assert not at.exception and not at.dataframe and not at.get('plotly_chart') and not downloads
    assert len(calls)==1  # ID changes suppress old person outputs, never auto-recalibrate.


def test_comparison_requires_matched_solver_and_bad_settings_fail_closed(actual_fits,monkeypatch):
    at,calls,downloads,_=app_with_real_fits(actual_fits,monkeypatch)
    at.button(key='python_mirt_start_Rasch').click().run()
    at.button(key='python_irt_2PL').click().run()
    at.button(key='python_mirt_start_2PL').click().run()
    assert not at.exception and len(calls)==2
    comparison=pd.read_csv(io.BytesIO(downloads['python_mirt_compare_csv_2PL']))
    assert set(comparison.Model)=={'Rasch','2PL'}
    for model in comparison.Model:
        assert comparison.set_index('Model').loc[model,'LogLik']==pytest.approx(actual_fits[1][model,21].fit.log_likelihood,abs=1e-10)
    before=downloads['python_mirt_persons_csv_2PL']
    downloads.clear();at.checkbox(key='python_mirt_output_2PL_comparison').uncheck().run()
    assert 'python_mirt_compare_csv_2PL' not in downloads and downloads['python_mirt_persons_csv_2PL']==before and len(calls)==2
    at.checkbox(key='python_mirt_output_2PL_comparison').check().run()
    at.checkbox(key='python_mirt_solver_advanced_2PL').check().run()
    assert len(calls)==2  # opt-in alone keeps numerically identical defaults valid.
    at.number_input(key='python_mirt_solver_2PL_n_quadpts').set_value(31).run()
    downloads.clear();at.button(key='python_mirt_start_2PL').click().run()
    assert not at.exception and len(calls)==3 and 'python_mirt_compare_csv_2PL' not in downloads
    assert at.session_state['python_mirt_fit_2PL']['outcome'].solver_settings['n_quadpts']==31
    at.number_input(key='python_mirt_solver_2PL_tol').set_value(0.).run()
    downloads.clear();at.button(key='python_mirt_start_2PL').click().run()
    assert not at.exception and any('tol must be finite and positive' in e.value for e in at.error)
    assert not downloads and not at.dataframe and 'python_mirt_fit_2PL' not in at.session_state
