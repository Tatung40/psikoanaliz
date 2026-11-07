import io
import zipfile

import pandas as pd
from docx import Document
from streamlit.testing.v1 import AppTest

from core import generalizability_advanced as ga
from tests.test_generalizability_ui import app, PAGE, ROOT


def test_engine_change_invalidates_all_result_downloads_and_missing_r_fails_closed(monkeypatch):
    at=app()
    at.selectbox(key='gt_engine').set_value('REML').run()
    assert not at.exception and 'gt_result' not in at.session_state
    assert not at.get('download_button')
    monkeypatch.setattr(ga.shutil,'which',lambda _:None)
    at.button(key='gt_run').click().run()
    assert not at.exception and at.error and 'Rscript' in at.error[0].value
    assert 'gt_result' not in at.session_state and not at.get('download_button')


def test_actual_multivariate_word_python_exports_and_stale_data():
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60)
    at.session_state['raw_df']=ga.example('multivariate')
    at.session_state['nav_page']=PAGE
    at.run()
    at.radio(key='gt_format').set_value('Çok değişkenli (uzun)').run()
    at.checkbox(key='gt_mv_enable').set_value(True).run()
    at.button(key='gt_mv_run').click().run()
    assert not at.exception and not at.error
    assert 'gt_mv_result' in at.session_state
    keys={b.key for b in at.get('download_button')}
    assert {'gt_mv_xlsx','gt_mv_zip','academic_word','repro_py','repro_zip'}<=keys
    history=at.session_state['_academic_history'][PAGE]
    doc=Document(io.BytesIO(history['docx']))
    assert doc.tables and any('kovaryans' in p.text.lower() for p in doc.paragraphs)
    before=at.session_state['gt_mv_result']['coefficients']['composite'].copy()
    at.run();pd.testing.assert_frame_equal(before,at.session_state['gt_mv_result']['coefficients']['composite'])
    changed=at.session_state['raw_df'].copy();changed.loc[0,'score']+=.1
    at.session_state['raw_df']=changed;at.run()
    assert not at.exception and 'gt_mv_result' not in at.session_state
    assert not {b.key for b in at.get('download_button')}&{'gt_mv_xlsx','gt_mv_zip','academic_word','repro_py','repro_zip'}


def test_d_optimization_downloads_and_settings_invalidation():
    at=app()
    at.button(key='gt_search_run').click().run()
    assert not at.exception and not at.error
    assert {'gt_optimization_xlsx','gt_optimization_zip'}<={b.key for b in at.get('download_button')}
    at.number_input(key='gt_search_cost').set_value(2.).run()
    assert not at.exception and 'gt_search_result' not in at.session_state
    assert not {b.key for b in at.get('download_button')}&{'gt_optimization_xlsx','gt_optimization_zip'}
