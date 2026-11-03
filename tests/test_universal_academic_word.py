"""Actual screen outputs, DOCX numbers and fail-closed input changes; no R or Monte Carlo."""
import io
from pathlib import Path
import pandas as pd
import pytest
from docx import Document
from streamlit.testing.v1 import AppTest
from tests.test_reproducible_page_reruns import _state,CASES

ROOT=Path(__file__).resolve().parents[1]
CTT=['3B. Betimsel İstatistikler','4. Test İstatistikleri','5. Madde Analizi',
     '6. Çeldirici Analizi','7. Güvenirlik & Ölçme Hatası','8. Grup Bazlı Analizler',
     '9. Bilimsel Grafikler','KTK: Çift Motor Rapor']
SCREENS=[(p,'abc_108',None,'ctt_double_run' if p.startswith('KTK:') else None) for p in CTT]
SCREENS += [(f'MTK: Python mirt {m}',f'irt_{i:03d}',None,f'python_mirt_start_{m}')
            for m,i in [('Rasch',0),('1PL',3),('2PL',6),('3PL',9),('4PL',12),('GRM',15),('PCM',18),('GPCM',21),('NRM',24)]]
SCREENS += [(p,d,a,b) for p,d,a,b,_ in CASES]
SCREENS += [('MTK: 2PL','irt_006',None,'py_mirt_run_2PL'),('MTK: 3PL','irt_009',None,'py_mirt_run_3PL')]
SCREENS += [('MTK: RSM','irt_027',None,'rsm_run'),('MTK: Çok Kategorili B14','irt_015',None,'b14_run')]


def _assert_word(at,page):
    assert not at.exception,[e.value for e in at.exception]
    history=at.session_state['_academic_history']
    assert page in history,[x.value for x in at.warning]+[x.value for x in at.caption]
    if 'legacy_docx' in history[page]:raw=history[page]['legacy_docx'][0]
    else:
        assert any(d.key=='academic_word' for d in at.get('download_button'))
        raw=history[page]['docx']
    doc=Document(raw if hasattr(raw,'read') else io.BytesIO(raw));text='\n'.join(p.text for p in doc.paragraphs)
    assert doc.tables or doc.inline_shapes
    assert 'NOT IMPLIED' in text or 'legacy_docx' in history[page]
    if 'result' in history[page]:
        result=history[page]['result']
        cells=' '.join(c.text for t in doc.tables for row in t.rows for c in row.cells)
        # Actual captured numeric cells, using the report engine's declared precision.
        from reports.academic_docx import _value
        numbers=[v for rows in result['tables'].values() for row in rows[:100] for v in row.values()
                 if isinstance(v,(float,int)) and not isinstance(v,bool)]
        assert all(_value(n) in cells for n in numbers)
        assert result['sample']['n']>0 and result['model']==page


@pytest.mark.parametrize('page,dataset,ack,button',SCREENS,ids=[x[0] for x in SCREENS])
def test_actual_family_word_and_python(page,dataset,ack,button):
    if dataset=='abc_108':
        from core import example_datasets as E
        dataset=next(s.id for s in E.registry().values() if s.family=='abc' and s.level==0)
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=600)
    for k,v in _state(dataset,page).items():at.session_state[k]=v
    at.run()
    if ack:at.checkbox(key=ack).check().run()
    if button:at.button(key=button).click().run()
    _assert_word(at,page)
    manifest=at.session_state['_repro_history'].manifest()
    assert manifest['status']=='GENERATED / UNVERIFIED',manifest['unsupported']
    code=[c.value for c in at.code if c.value.startswith('# PsikoAnaliz reproducible analysis')]
    assert len(code)==1;compile(code[0],'analysis.py','exec')
    at.run();_assert_word(at,page)


def test_real_gt_word_and_missing_data_word():
    from tests.test_generalizability_ui import app,PAGE
    at=app();_assert_word(at,PAGE)
    at=AppTest.from_file(str(ROOT/'pages/90_Kayip_Veri_Analizi_ve_Atama.py'),default_timeout=60)
    x=pd.read_csv(ROOT/'tests/fixtures/irt_b10/data_irt2pl_ok.csv').iloc[:40,:4].astype(float)
    x.iloc[0,0]=float('nan');at.session_state['raw_df']=x
    at.run();_assert_word(at,'Eksik Veri ve Atama')


def test_center_word_and_comparison_word():
    from tests.test_dif_center_ui import app
    at=app();at.button(key='dc_run').click().run();_assert_word(at,'MTK: DIF Analiz Merkezi')
    page='MTK: Karşılaştırma';at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=180)
    for k,v in _state('irt_006',page).items():at.session_state[k]=v
    for model in ('1PL','2PL'):
        at.session_state['nav_page']='MTK: '+model
        at.run();at.button(key='py_mirt_run_'+model).click().run()
    at.session_state['nav_page']=page;at.run();_assert_word(at,page)
    assert at.session_state['_repro_history'].manifest()['status']=='GENERATED / UNVERIFIED'


def test_bootstrap_word_is_descriptive_while_python_inference_stays_unsupported():
    page='MTK: 3PL/4PL Bootstrap DIF';at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=600)
    for k,v in _state('dif_075',page).items():at.session_state[k]=v
    at.run();at.button(key='b34_run').click().run();assert not at.exception
    _assert_word(at,page)
    manifest=at.session_state['_repro_history'].manifest()
    assert manifest['status']=='UNSUPPORTED'
    assert any('bootstrap DIF inference is gated' in x for x in manifest['unsupported'])
    assert not {d.key for d in at.get('download_button')} & {'repro_py','repro_zip'}


@pytest.mark.parametrize('change',['values','settings'])
def test_cached_ctt_word_and_python_fail_closed_without_recalculation(change):
    # The old dual page retains its raw result internally; the common exports must
    # reject it even before that legacy page gains its own invalidation mechanism.
    from core import example_datasets as E
    dataset=next(s.id for s in E.registry().values() if s.family=='abc' and s.level==0)
    page='KTK: Çift Motor Rapor';at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90)
    for k,v in _state(dataset,page).items():at.session_state[k]=v
    at.run();at.button(key='ctt_double_run').click().run();_assert_word(at,page)
    if change=='values':
        x=at.session_state['scored_df'].copy();c=at.session_state['item_cols'][0];x.loc[x.index[0],c]=1-x.loc[x.index[0],c];at.session_state['scored_df']=x
    else:at.multiselect(key='ctt_double_columns').set_value(at.session_state['item_cols'][:-1])
    at.run();assert not at.exception
    keys={d.key for d in at.get('download_button')}
    assert not keys & {'academic_word','ctt_double_docx','repro_py','repro_zip'}
    assert at.session_state['_repro_history'].manifest()['status']=='UNSUPPORTED'
