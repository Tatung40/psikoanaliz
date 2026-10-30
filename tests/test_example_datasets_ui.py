from pathlib import Path
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from core import example_datasets as E

ROOT=Path(__file__).resolve().parents[1]


def app(page='2. Veri Yükleme'):
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90)
    at.session_state['nav_page']=page
    return at.run()


def choose(at,page,d):
    at.selectbox(key='examples_pick_'+page).set_value(d.id).run()
    assert not at.exception
    assert not any(b.key=='examples_activate_'+page for b in at.button)
    assert not at.exception
    return at


@pytest.mark.parametrize('design',list(E.DESIGNS))
def test_g_example_loads_maps_and_runs_real_g_d(design):
    d=next(d for d in E.registry().values() if d.family=='g' and d.design==design and d.level==0)
    at=app(E.G_PAGE)
    choose(at,E.G_PAGE,d)
    assert not at.error
    assert at.session_state['gt_result']['settings']['design']==design
    assert len(at.session_state['gt_result']['d']) in (2,4)
    assert at.session_state['nav_page']==E.G_PAGE
    assert {b.key for b in at.get('download_button')} >= {'gt_csv','gt_xlsx','gt_png','gt_zip'}
    at.run();assert not at.exception and 'gt_result' in at.session_state


def test_abc_20_loads_scores_manual_key_change_clears_all_results():
    page='3. Puanlama & Veri Hazırlama';d=next(d for d in E.registry().values() if d.family=='abc' and d.k==20)
    at=app(page);choose(at,page,d)
    assert not at.error
    assert len(at.session_state['answer_key'])==20
    assert at.session_state['scored_df'] is not None
    at.session_state['gt_result']={'stale':True}
    item=at.session_state['item_cols'][0]
    old=at.session_state['answer_key'][item]
    at.text_input(key='key_'+item).set_value('E' if old!='E' else 'A').run()
    assert not at.exception
    assert at.session_state['scored_df'] is None and 'gt_result' not in at.session_state
    next(b for b in at.button if b.label=='🎯 Veriyi Cevap Anahtarına Göre Puanla').click().run()
    assert not at.exception and not at.error and at.session_state['scored_df'] is not None
    assert at.session_state['group_col'] in at.session_state['scored_df']


@pytest.mark.parametrize('page,label_key',[
    ('MTK: Python mirt NRM','python_mirt_nominal_labels'),
    ('MTK: Python mirt DIF','pmd_labels'),
    (E.M_PAGE,'md_labels'),
])
def test_new_example_clears_previous_models_keys_and_downloads(page,label_key):
    at=app(page)
    at.session_state['answer_key']={'OLD':'A'}
    at.session_state['python_mirt_cache']={'old':True}
    at.session_state['scored_df']=pd.DataFrame({'OLD':[1,1,1]})
    d=next(d for d in E.available(page) if d.model=='NRM');choose(at,page,d)
    assert not at.error
    assert at.session_state['answer_key']=={}
    assert 'python_mirt_cache' not in at.session_state
    assert at.text_input(key=label_key).value=='A,B,C,D'
    assert not any(b.key=='python_mirt_parameters_csv_NRM' for b in at.get('download_button'))


def test_guides_visible_without_loading_data_and_no_fake_calibration():
    for page in ('MTK: Python mirt GRM',E.M_PAGE,'MTK: Python mirt DIF','Boyutluluk ve Faktör Analizi'):
        at=app(page)
        assert not at.exception
        assert any(e.label=='Kendi Veri Setimi Nasıl Hazırlamalıyım?' for e in at.expander)
        assert at.session_state['raw_df'] is None


def test_missing_page_uses_example_and_invalidates_changed_data():
    page=E.MISSING_PAGE
    at=AppTest.from_file(str(ROOT/'pages/90_Kayip_Veri_Analizi_ve_Atama.py'),default_timeout=90).run()
    choose(at,page,E.available(page)[0])
    assert not at.error
    next(b for b in at.button if b.label=='▶ Seçili yöntemi uygula').click().run()
    assert not at.exception and 'missing_lab_result' in at.session_state
    d=at.session_state['raw_df'].copy();d.iloc[0,1]=1-d.iloc[0,1]
    at.session_state['raw_df']=d;at.run()
    assert not at.exception and 'missing_lab_result' not in at.session_state


def test_own_file_callback_clears_example_and_old_analyses(monkeypatch):
    from ui import example_datasets_view as view
    uploaded=object()
    state={'nav_page':'2. Veri Yükleme','privacy_upload_nonce':2,'private_file_upload_2':uploaded,
           'raw_df':pd.DataFrame({'OLD':[1]}),'answer_key':{'OLD':'A'},'gt_result':{'old':True},
           '_example_active':'OLD','md_model':'NRM'}
    monkeypatch.setattr(view.st,'session_state',state)
    view.own_file_changed('private_file_upload_2')
    assert state=={'nav_page':'2. Veri Yükleme','privacy_upload_nonce':2,'private_file_upload_2':uploaded}
