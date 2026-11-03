"""Real fits and captured-screen replay; never serialize opaque fits as invented inputs."""
from pathlib import Path
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]


def test_returning_to_fitted_page_keeps_its_actual_producer():
    data=pd.read_csv(ROOT/'tests/fixtures/irt_b10/data_irt2pl_ok.csv').iloc[:200,:6].astype(int)
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=180)
    for k,v in dict(raw_df=data,scored_df=data.copy(),item_cols=list(data),id_col=None,
                    group_col=None,is_dichotomous=True,nav_page='MTK: Python mirt 2PL').items():at.session_state[k]=v
    at.run();at.button(key='python_mirt_start_2PL').click().run()
    assert not at.exception
    before=at.session_state['_repro_history'].manifest()
    assert before['status']=='GENERATED / UNVERIFIED',before['unsupported']
    outcome=at.session_state['python_mirt_fit_2PL']['outcome']
    at.session_state['nav_page']='3B. Betimsel İstatistikler';at.run()
    at.session_state['nav_page']='MTK: Python mirt 2PL';at.run()
    assert not at.exception
    after=at.session_state['_repro_history'].manifest()
    assert after['status']=='GENERATED / UNVERIFIED',after['unsupported']
    assert at.session_state['python_mirt_fit_2PL']['outcome'] is outcome
    # A pre-deploy session can contain the opaque result but no producing recipe.
    # It must request a real rerun, never invent a serializer or export old Word.
    for key in ('_repro_history','_repro_pages','_academic_history'):del at.session_state[key]
    at.run();assert not at.exception
    assert at.session_state['_repro_history'].manifest()['status']=='UNSUPPORTED'
    assert not {d.key for d in at.get('download_button')} & {'repro_py','repro_zip','python_mirt_2pl_academic_word','academic_word'}


def test_optional_multistart_records_the_actual_selected_producer():
    from tests.test_reproducible_page_reruns import _state
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=300)
    for k,v in _state('irt_009','MTK: Python mirt 3PL').items():at.session_state[k]=v
    at.run();at.checkbox(key='python_mirt_multistart_3PL').check().run()
    at.button(key='python_mirt_start_3PL').click().run();assert not at.exception
    for _ in range(2):
        recorder=at.session_state['_repro_history'];manifest=recorder.manifest()
        assert manifest['status']=='GENERATED / UNVERIFIED',manifest['unsupported']
        assert len([c for c in manifest['calls'] if c['function']=='multistart'])==1
        assert len(at.session_state['python_mirt_fit_3PL']['multistart'][0])==8
        compile(next(c.value for c in at.code if c.value.startswith('# PsikoAnaliz reproducible analysis')),'analysis.py','exec')
        at.run()


def test_chart_zip_is_portable_and_preserves_actual_observations(tmp_path):
    import io,json,os,subprocess,sys,zipfile
    from core.reproducible import finish,public_call
    from ui.reproducible_capture import begin,replay_manifest,python_code,bundle
    begin('9. Bilimsel Grafikler')
    scores=pd.read_csv(ROOT/'tests/fixtures/irt_b10/data_irt2pl_ok.csv').iloc[:40,:4].sum(axis=1)
    public_call('ui.charts','plot_score_histogram',scores,bins=12)
    recorder=finish();manifest=replay_manifest(recorder);code=python_code(manifest)
    assert 'ui/charts.py' in manifest['runtime_sha256']
    with zipfile.ZipFile(io.BytesIO(bundle(manifest,code))) as z:z.extractall(tmp_path)
    (tmp_path/'analysis.py').write_text(code+'\njson.dump(list(results[0].data[0].x),open("observed.json","w"))\n')
    run=subprocess.run([sys.executable,'analysis.py'],cwd=tmp_path,env=dict(os.environ,PYTHONPATH=''),capture_output=True,text=True,timeout=60)
    assert run.returncode==0,run.stderr[-2000:]
    assert json.loads((tmp_path/'observed.json').read_text())==scores.tolist()


def test_graphic_selection_updates_word_without_refitting():
    import io,zipfile
    from tests.test_reproducible_page_reruns import _state
    page='MTK: Python mirt 1PL';at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=180)
    for k,v in _state('irt_003',page).items():at.session_state[k]=v
    at.run()
    assert not any(d.key=='academic_word' for d in at.get('download_button'))
    at.button(key='python_mirt_start_1PL').click().run()
    before=at.session_state['_academic_history'][page]['docx']
    fitted=at.session_state['python_mirt_fit_1PL']['outcome']
    widget=at.selectbox(key='python_mirt_plot_item_1PL');widget.set_value(widget.options[1]).run()
    assert not at.exception
    after=at.session_state['_academic_history'][page]['docx']
    def media(raw):
        with zipfile.ZipFile(io.BytesIO(raw)) as z:return [z.read(n) for n in z.namelist() if n.startswith('word/media/')]
    assert media(before)!=media(after)
    assert at.session_state['python_mirt_fit_1PL']['outcome'] is fitted
    assert at.session_state['_repro_history'].manifest()['status']=='GENERATED / UNVERIFIED'
    at.run();assert at.session_state['_academic_history'][page]['docx'] is after
