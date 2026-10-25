from pathlib import Path
import pytest

@pytest.mark.parametrize('model',['GRM','NRM','3PL','2PL'])
def test_python_engine_invalidation_includes_categories(model,monkeypatch):
    """Exercise cache eviction; do not depend on private variable names."""
    import pandas as pd
    from streamlit.testing.v1 import AppTest
    import ui.irt_python_library_view as view
    def no_stale_report(*args,**kwargs):
        pytest.fail('Stale cached outcome was accessed')
    monkeypatch.setattr(view,'build_report',no_stale_report)
    source="""
import streamlit as st
import pandas as pd
from ui.irt_python_library_view import render
model=st.session_state['model']
x=pd.DataFrame({f'M{i}':list(range(4))*10 for i in range(4)})
if model in ('2PL','3PL'):x=x%2
if model=='NRM':x=x.replace(dict(enumerate('ABCD')))
render(x,items=list(x),model=model)
"""
    at=AppTest.from_string(source,default_timeout=30)
    at.session_state['model']=model
    at.run();assert not at.exception
    x=pd.DataFrame({f'M{i}':list(range(4))*10 for i in range(4)})
    if model in ('2PL','3PL'):x=x%2
    labels=tuple('ABCD') if model=='NRM' else tuple(range(4)) if model=='GRM' else ()
    key=(tuple(x),view._data_fingerprint(x),4 if labels else None,labels,
         'MAP' if model=='3PL' else 'ML','nine-model-report-v1')
    at.session_state[f'python_mirt_fit_{model}']={'key':key,'outcome':object()}
    if model=='GRM':at.number_input(key='python_mirt_categories_GRM').set_value(5).run()
    elif model=='NRM':at.text_input(key='python_mirt_nominal_labels').set_value('B,A,C,D').run()
    elif model=='3PL':at.radio(key='python_mirt_estimation_3PL').set_value('ML').run()
    else:at.multiselect(key='python_mirt_cols_2PL').set_value(list(x)[:3]).run()
    assert not at.exception
    assert f'python_mirt_fit_{model}' not in at.session_state
    assert not at.dataframe and not at.get('plotly_chart') and not at.get('download_button')

def test_homepage_does_not_falsely_hide_irt():
    source=Path("app.py").read_text(encoding="utf-8")
    assert "Bu analizler mevcut sürümde henüz kullanıma sunulmamıştır." not in source
    assert "4PL formel" in source
