from pathlib import Path
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]
PAGE='Genellenebilirlik Kuramı (G ve D Çalışmaları)'


def app():
    d=pd.read_csv(ROOT/'validation/generalizability/balanced.csv')
    d=d[d.r==0].drop(columns='r').reset_index(drop=True)
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60)
    at.session_state['raw_df']=d
    at.session_state['nav_page']=PAGE
    at.run()
    assert not at.exception
    at.button(key='gt_run').click().run()
    assert not at.exception and not at.error
    return at


@pytest.mark.parametrize('change',['values','design','column','score','scenario','target','invalid'])
def test_stale_all_downloads_and_results_suppressed(change):
    at=app()
    assert len(at.get('download_button')) >= 4
    assert 'gt_result' in at.session_state
    if change=='values':
        d=at.session_state['raw_df'].copy(); d.loc[0,'score']+=.5
        at.session_state['raw_df']=d
    if change=='design':at.selectbox(key='gt_design').set_value('p x r')
    if change=='column':at.selectbox(key='gt_col_i').set_value('p')
    if change=='score':at.selectbox(key='gt_score').set_value('i')
    if change=='scenario':at.text_input(key='gt_d_p x i_i').set_value('3,4')
    if change=='target':at.number_input(key='gt_target').set_value(.9)
    if change=='invalid':at.text_input(key='gt_d_p x i_i').set_value('bad')
    at.run()
    assert not at.exception
    assert 'gt_result' not in at.session_state
    assert not at.get('download_button')


def test_unchanged_results_persist_navigation_and_python_export():
    at=app()
    assert any(b.key=='nav_generalizability' for b in at.button)
    assert any(t.label=='</> Python Kodları' for t in at.tabs)
    assert {b.key for b in at.get('download_button')} >= {'gt_xlsx','gt_zip','gt_csv','gt_png','repro_py','repro_zip'}
    result=at.session_state['gt_result']
    at.run()
    assert not at.exception and not at.error
    assert at.session_state['nav_page']==PAGE
    pd.testing.assert_frame_equal(at.session_state['gt_result']['g'],result['g'])
    assert {b.key for b in at.get('download_button')} >= {'gt_xlsx','gt_zip','gt_csv','gt_png','repro_py','repro_zip'}


def test_wide_input_actual_library_run():
    at=app()
    d=at.session_state['raw_df'].pivot(index='p',columns='i',values='score').reset_index()
    d.columns=['person','A','B','C']
    at.session_state['raw_df']=d
    at.session_state['item_cols']=['A','B','C']
    at.radio(key='gt_format').set_value('Geniş (birey × madde)').run()
    at.button(key='gt_run').click().run()
    assert not at.exception and not at.error
    assert at.session_state['gt_result']['settings']['design']=='p x i'
