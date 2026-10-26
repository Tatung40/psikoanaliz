"""Reuse real PCM calibration for service, import/export and Streamlit checks."""
import io
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from core.irt_b14 import fit_b14

@pytest.fixture(scope='module')
def pcm():
    data=pd.read_csv('validation/evidence/p5_browser/poly_ordinal.csv')
    return data,fit_b14(data,'PCM')


def test_actual_pcm_service_reports_and_import_export(pcm):
    from core.data_processing import load_dataset as load_data
    from core.irt_b14_persons_info import person_scores,information_table,independent_information
    from core.irt_b14_model_curves import fitted_category_curves
    from reports.irt_b14_report import report_tables,export_xlsx
    from reports.irt_b14_narrative import guided_report
    data,result=pcm;fit=result['result']
    assert fit.converged and fit.n_parameters==data.shape[1]*3
    np.testing.assert_array_equal(fit.model.discrimination,np.ones(data.shape[1]))
    for fmt in ('csv','xlsx'):
        buf=io.BytesIO()
        if fmt=='csv':buf.write(data.to_csv(index=False).encode())
        else:data.to_excel(buf,index=False)
        imported,_=load_data(buf.getvalue(),fmt)
        pd.testing.assert_frame_equal(imported,data)
    persons=person_scores(result);assert len(persons)==len(data)
    assert np.isfinite(persons['θ']).all() and (persons['SH(θ)']>0).all()
    theta=np.linspace(-3,3,13);info=information_table(result,theta)
    np.testing.assert_allclose(info.filter(like='I_'),independent_information(result,theta),rtol=1e-4,atol=1e-8)
    np.testing.assert_allclose(info['SEM(θ)']**-2,info['Test bilgisi'])
    curves=fitted_category_curves(result,theta)
    np.testing.assert_allclose(curves.groupby(['Madde','Theta']).Olasılık.sum(),1,atol=1e-12)
    from core.irt_b14_diagnostics import model_diagnostics
    status,diagnostics=model_diagnostics(result)
    assert len(status)==2 and diagnostics, status.to_dict('records')
    tables=report_tables(result);meta=tables['Meta'].set_index('Özellik')['Değer']
    assert meta['AIC']==fit.aic and meta['BIC']==fit.bic and meta['Yakınsama']
    book=pd.read_excel(io.BytesIO(export_xlsx(tables)),sheet_name=None)
    assert {'Motor Parametreleri','Kişi Yetenekleri','Bilgi Fonksiyonları'}<=set(book)
    assert 'a=1' in guided_report(result) and 'sıralı olmak zorunda değildir' in guided_report(result)


def test_pcm_ui_reuses_fit_invalidates_diagnostics_zip_and_failed_retry(pcm,monkeypatch):
    from streamlit.testing.v1 import AppTest
    import ui.irt_b14_view as view
    data,result=pcm;calls=[]
    def fit(*a,**k):calls.append(1);return result
    monkeypatch.setattr(view,'fit_b14',fit)
    at=AppTest.from_file(str(Path('app.py').resolve()),default_timeout=120)
    for k,v in dict(raw_df=data,scored_df=None,item_cols=list(data),id_col=None,group_col=None,nav_page='MTK: Çok Kategorili B14').items():at.session_state[k]=v
    at.run();at.radio(key='b14_model').set_value('PCM').run()
    at.button(key='b14_run').click().run();assert not at.exception
    assert at.session_state['b14_current']['result']['model']=='PCM' and len(calls)==1
    assert len(at.get('download_button'))>=6 and at.get('plotly_chart')
    at.run();assert len(calls)==1
    from copy import copy
    nonconverged=copy(result['result']);nonconverged.converged=False
    at.session_state['b14_current']['result']=dict(result,result=nonconverged)
    at.run();assert any('yakınsamadı' in e.value for e in at.error)
    at.session_state['b14_current']['result']=result
    for key in ('b14_zip','b14_diagnostics','b14_compare'):at.session_state[key]='stale sentinel'
    changed=data.copy();changed.iloc[0,0]=(changed.iloc[0,0]+1)%4
    at.session_state['raw_df']=changed;at.run();assert not at.exception
    for key in ('b14_current','b14_zip','b14_diagnostics','b14_compare'):assert key not in at.session_state
    assert not any(getattr(e.proto,'label','')=='B14 tüm tablolar Excel' for e in at.get('download_button'))
    def fail(*a,**k):raise RuntimeError('controlled PCM failure')
    monkeypatch.setattr(view,'fit_b14',fail)
    at.button(key='b14_run').click().run()
    assert not at.exception and any('controlled PCM failure' in e.value for e in at.error)
    assert 'b14_current' not in at.session_state


@pytest.mark.parametrize('kind',['missing','fraction','negative','constant','category_gap'])
def test_pcm_invalid_categories_stop_before_calibration(kind,monkeypatch):
    import mirt
    data=pd.read_csv('validation/evidence/p5_browser/poly_ordinal.csv').astype(float)
    if kind=='missing':data.iloc[0,0]=np.nan
    elif kind=='fraction':data.iloc[0,0]=.5
    elif kind=='negative':data.iloc[0,0]=-1
    elif kind=='constant':data.iloc[:,0]=0
    else:data=data.replace(2,3)
    monkeypatch.setattr(mirt,'fit_mirt',lambda *a,**k:pytest.fail('Invalid data reached calibration'))
    with pytest.raises(ValueError):fit_b14(data,'PCM')
