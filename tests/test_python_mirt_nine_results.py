"""Real nine-family calibration -> reporting -> Streamlit; one fit/family per session."""
import io
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from core.irt_python_mirt_adapter import fit_python_mirt
from core.irt_python_mirt_report import MODELS, BINARY, build_report
from ui.irt_python_library_view import _csv


@pytest.fixture(scope='module', params=MODELS)
def calibrated(request):
    model=request.param
    if model in BINARY:
        path=(f'tests/fixtures/irt_b11/data_irt{model.lower()}_ok.csv' if model in ('3PL','4PL')
              else 'tests/fixtures/irt_b10/data_irt1pl_ok.csv')
    else:path='validation/evidence/p5_browser/poly_ordinal.csv'
    frame=pd.read_csv(path).iloc[:300,:6].astype(int)
    outcome=fit_python_mirt(frame.to_numpy(),model='1PL' if model=='Rasch' else model,
        n_categories=None if model in BINARY else 4,
        estimation='MAP' if model in ('3PL','4PL') else 'ML',common_discrimination=model=='1PL')
    labels=() if model in BINARY else tuple('ABCD') if model=='NRM' else tuple(range(4))
    report=build_report(outcome,model=model,items=frame.columns,labels=labels)
    request.config.cache.set(f'nine_results/{model}',dict(report.metadata))
    return model,frame,outcome,labels,report


def test_actual_estimates_curves_information_csv(calibrated):
    model,frame,outcome,labels,report=calibrated
    assert not report.errors,report.errors
    assert outcome.fit.converged, 'Fixture calibration did not converge; not a successful fit'
    assert report.metadata['Yakınsama']==bool(outcome.fit.converged)
    assert report.metadata['LogLik']==outcome.fit.log_likelihood
    assert len(report.parameters)==len(frame.columns)*(4 if model=='NRM' else 1)
    assert report.parameters.Madde.drop_duplicates().tolist()==list(frame.columns)
    assert not report.curves.empty and not report.information.empty
    theta=report.information['θ'].to_numpy()
    actual=np.asarray(outcome.fit.model.probability(theta[:,None]))
    for j,item in enumerate(frame.columns):
        p=report.curves[report.curves.Madde==item].pivot(index='Theta',columns='Kategori',values='Olasılık').to_numpy()
        expected=np.column_stack([1-actual[:,j],actual[:,j]]) if model in BINARY else actual[:,j,:]
        np.testing.assert_allclose(p,expected,atol=1e-12)
        np.testing.assert_allclose(report.information[f'I_{item}'],np.asarray(outcome.fit.model.information(theta[:,None],item_idx=j)).ravel(),atol=1e-10)
    for table in (report.parameters,report.curves,report.information,pd.DataFrame([report.metadata])):
        exported=pd.read_csv(io.BytesIO(_csv(table)),dtype={c:str for c in table if pd.api.types.is_string_dtype(table[c])})
        pd.testing.assert_frame_equal(exported,table,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
    native=outcome.fit.model.parameters
    if model in BINARY:
        for source,target in [('discrimination','a'),('difficulty','b'),('guessing','c'),('upper','d')]:
            if source in native:np.testing.assert_allclose(report.parameters[target],native[source])
        if model=='Rasch':assert (report.parameters.a==1).all()
        if model=='1PL':
            assert report.parameters.a.nunique()==1
            assert outcome.fit.n_parameters==len(frame.columns)+1
    elif model=='NRM':
        np.testing.assert_array_equal(report.parameters['Eğim (A)'].to_numpy().reshape(6,4),native['slopes'])
        np.testing.assert_array_equal(report.parameters['Kesişim (D)'].to_numpy().reshape(6,4),native['intercepts'])
    else:
        np.testing.assert_array_equal(report.parameters.a,native['discrimination'])
        key='thresholds' if model=='GRM' else 'steps'
        np.testing.assert_array_equal(report.parameters.filter(regex=' b').to_numpy(),native[key])


def test_actual_streamlit_tables_plots_downloads_and_invalidation(calibrated,monkeypatch):
    from streamlit.testing.v1 import AppTest
    import ui.irt_python_library_view as view
    model,frame,outcome,labels,report=calibrated
    calls=[]
    rendered_figures=[]
    original_pyplot=view.st.pyplot
    def capture_pyplot(fig,*args,**kwargs):
        assert fig is not None and len(fig.axes)==1
        assert len(fig.axes[0].lines)>=1
        rendered_figures.append(fig)
        return original_pyplot(fig,*args,**kwargs)
    monkeypatch.setattr(view.st,'pyplot',capture_pyplot)
    def fitted(x,**kwargs):
        np.testing.assert_array_equal(x,frame.to_numpy())
        assert kwargs['common_discrimination']==(model=='1PL')
        calls.append(kwargs)
        return outcome
    monkeypatch.setattr(view,'fit_python_mirt',fitted)
    data=frame.replace(dict(enumerate(labels))) if model=='NRM' else frame
    at=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'),default_timeout=120)
    for k,v in dict(raw_df=data,scored_df=frame%2,item_cols=list(frame.columns),id_col=None,group_col=None,
                    is_dichotomous=model in BINARY,nav_page=f'MTK: Python mirt {model}').items():
        at.session_state[k]=frame if k=='scored_df' and model in BINARY else v
    at.run();assert not at.exception
    at.button(key=f'python_mirt_start_{model}').click().run()
    assert not at.exception
    assert len(at.dataframe)==4 and len(rendered_figures)==4
    assert len(at.get('download_button'))==7
    assert sum('Kestirim yakınsadı' in e.value for e in at.success)==int(bool(outcome.fit.converged))
    pd.testing.assert_frame_equal(at.dataframe[2].value,report.parameters)
    assert {'Birey','Ham puan','Yetenek (θ)','SH (θ)','Yöntem','Model'} <= set(at.dataframe[3].value.columns)
    assert len(at.dataframe[3].value)==len(frame)
    at.run();assert len(calls)==1 and len(at.dataframe)==4 and len(rendered_figures)==8
    # Nonconvergence is explicit in UI and exported parameters.
    from copy import copy
    failed=copy(outcome.fit);failed.converged=False
    at.session_state[f'python_mirt_fit_{model}']['outcome']=replace(outcome,fit=failed)
    at.run()
    assert any('Kestirim yakınsamadı' in e.value for e in at.error)
    assert not any('Kestirim yakınsadı' in e.value for e in at.success)
    assert len(at.dataframe)==2 and not at.dataframe[1].value.Yakınsama.any()
    at.session_state[f'python_mirt_fit_{model}']['outcome']=outcome
    changed=data.copy();changed.iloc[0,0]=labels[(frame.iloc[0,0]+1)%4] if model=='NRM' else (frame.iloc[0,0]+1)%(2 if model in BINARY else 4)
    at.session_state['raw_df']=changed;at.session_state['scored_df']=changed
    at.run();assert not at.exception
    assert not at.dataframe and not at.get('download_button')
    # Invalid input also clears every output rather than retaining the previous fit.
    at.session_state[f'python_mirt_fit_{model}']={'key':None,'outcome':outcome}
    at.multiselect(key=f'python_mirt_cols_{model}').set_value(list(frame.columns[:2])).run()
    assert at.error and not at.dataframe and f'python_mirt_fit_{model}' not in at.session_state


def test_missing_api_and_nonconvergence_are_not_success(calibrated):
    model,frame,outcome,labels,_=calibrated
    fit=outcome.fit
    fields={k:getattr(fit,k) for k in ('n_iterations','log_likelihood','aic','bic','n_parameters','n_observations','model')}
    proxy=SimpleNamespace(**fields,converged=False,coef=fit.coef)
    report=build_report(replace(outcome,fit=proxy),model=model,items=frame.columns,labels=labels)
    assert report.metadata['Yakınsama'] is False
    assert any('Yakınsama sağlanmadı' in w for w in report.warnings)
    assert not report.parameters.Yakınsama.any()
    missing=build_report(replace(outcome,fit=object()),model=model,items=frame.columns,labels=labels)
    assert missing.metadata['Yakınsama'] is None and missing.errors
    assert missing.parameters.empty and missing.curves.empty


@pytest.mark.parametrize('model',MODELS)
def test_engine_exception_cannot_leave_outputs(model,monkeypatch):
    from streamlit.testing.v1 import AppTest
    import ui.irt_python_library_view as view
    def fail(*args,**kwargs):raise RuntimeError('intentional calibration failure')
    monkeypatch.setattr(view,'fit_python_mirt',fail)
    source='''
import pandas as pd
from ui.irt_python_library_view import render
import streamlit as st
model=st.session_state['model']
x=pd.DataFrame({f'M{i}':list(range(4))*10 for i in range(3)})
if model in ('Rasch','1PL','2PL','3PL','4PL'):x=x%2
if model=='NRM':x=x.replace(dict(enumerate('ABCD')))
render(x,items=list(x),model=model)
'''
    at=AppTest.from_string(source,default_timeout=30)
    at.session_state['model']=model
    # Legacy/stale fit cannot survive a failed new request.
    at.session_state[f'python_mirt_fit_{model}']={'key':(), 'outcome':object()}
    at.run();at.button(key=f'python_mirt_start_{model}').click().run()
    assert not at.exception and any('intentional calibration failure' in e.value for e in at.error)
    assert not at.dataframe and not at.get('download_button')
    assert f'python_mirt_fit_{model}' not in at.session_state


@pytest.mark.parametrize('model,estimation',[('2PL','ML'),('1PL','MAP')])
def test_common_discrimination_requires_explicit_1pl_ml(model,estimation):
    with pytest.raises(ValueError,match='Common discrimination'):
        fit_python_mirt(np.ones((30,3)),model=model,estimation=estimation,common_discrimination=True)


@pytest.mark.parametrize('settings', [
    {'max_iter': 500, 'tol': 1e-4, 'n_quadpts': 21},
    {'max_iter': 650, 'tol': 1e-5, 'n_quadpts': 31},
])
def test_real_rasch_solver_settings_change_engine_configuration(settings):
    """Real-library calibration: settings must reach the public API and produce finite fits."""
    from unittest.mock import patch
    import mirt
    x=pd.read_csv('tests/fixtures/irt_b10/data_irt1pl_ok.csv').iloc[:150,:5].to_numpy(dtype=int)
    native=mirt.fit_mirt
    captured=[]
    def record(data, **kwargs):
        captured.append(dict(kwargs))
        return native(data, **kwargs)
    with patch('mirt.fit_mirt',side_effect=record):
        result=fit_python_mirt(x,model='Rasch',solver_settings=settings)
    assert len(captured)==1
    assert all(captured[0][name]==value for name,value in settings.items())
    assert np.isfinite(result.fit.log_likelihood)
    assert np.isfinite(result.fit.aic)
    assert result.fit.n_iterations <= settings['max_iter']


def test_real_rasch_wright_map_library_and_person_locations(calibrated):
    """Rasch map must use the library's native plotting API and real EAP estimates."""
    model,frame,outcome,labels,report=calibrated
    if model!='Rasch':
        pytest.skip('Wright Map gate currently enabled only for Rasch')
    assert outcome.fit.converged and not report.errors
    import mirt
    from core.irt_person_abilities import person_abilities
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    people=person_abilities(frame.to_numpy(),report.curves,model=model)
    theta=people['Yetenek (θ)'].to_numpy(dtype=float)
    assert len(theta)==len(frame) and np.isfinite(theta).all()
    fig,ax=plt.subplots()
    try:
        actual=mirt.plot_person_item_map(outcome.fit,theta,ax=ax)
        assert actual is ax
        assert len(fig.axes)>=2
        locations=np.asarray(outcome.fit.model.parameters['difficulty'],dtype=float)
        assert len(locations)==len(frame.columns)
        plotted=fig.axes[1].collections[0].get_offsets()[:,1]
        np.testing.assert_allclose(plotted,locations,atol=1e-12)
    finally:
        plt.close(fig)
