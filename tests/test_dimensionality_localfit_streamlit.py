"""AppTest: iki yeni ekran çalışır, analiz eder ve indirme düğmeleri üretir."""
from pathlib import Path
import numpy as np
import pandas as pd
from streamlit.testing import v1 as st_testing

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _app(df, page):
    at = st_testing.AppTest.from_file(APP, default_timeout=180)
    at.session_state["raw_df"] = df
    at.session_state["scored_df"] = df.copy()
    at.session_state["item_cols"] = list(df.columns)
    at.session_state["id_col"] = None
    at.session_state["group_col"] = None
    at.session_state["is_dichotomous"] = True
    at.session_state["nav_page"] = page
    return at


def _data(n=500, k=8, seed=2):
    rng = np.random.default_rng(seed)
    th = rng.normal(size=n)
    p = 1 / (1 + np.exp(-1.5 * (th[:, None] - np.linspace(-1, 1, k))))
    return pd.DataFrame((rng.random(p.shape) < p).astype(float), columns=[f"M{i+1}" for i in range(k)])


def _run(page, run_key):
    at = _app(_data(), page)
    at.run(); assert not at.exception
    assert any(b.key in ("nav_dimensionality", "nav_local_fit") for b in at.button)
    next(b for b in at.button if b.key == run_key).click()
    if run_key == "dim_run":
        at.number_input(key="dim_reps").set_value(10)
    at.run(); assert not at.exception
    assert not at.error
    assert len(at.get('download_button')) >= 5
    prefix='dim' if run_key=='dim_run' else 'lf'
    result=at.session_state[prefix+'_result'];key=at.session_state[prefix+'_key']
    original=at.session_state['scored_df'].copy()
    changed=original.copy();changed.iloc[0,0]=1-changed.iloc[0,0]
    at.session_state['scored_df']=changed;at.run()
    assert not at.exception and not at.get('download_button') and prefix+'_result' not in at.session_state
    at.session_state['scored_df']=original
    at.session_state[prefix+'_result']=result;at.session_state[prefix+'_key']=key
    at.run();assert len(at.get('download_button')) >= 5
    if prefix=='dim':at.selectbox(key='dim_rot').set_value('varimax').run()
    else:at.selectbox(key='lf_model').set_value('3PL').run()
    assert not at.exception and not at.get('download_button') and prefix+'_result' not in at.session_state
    # Restore a valid cached result for the caller's existing output assertions.
    if prefix=='dim':at.selectbox(key='dim_rot').set_value('promax')
    else:at.selectbox(key='lf_model').set_value('2PL')
    at.session_state[prefix+'_result']=result;at.session_state[prefix+'_key']=key
    at.run()
    return at


def test_dimensionality_page():
    at = _run("Boyutluluk ve Faktör Analizi", "dim_run")
    assert any("Paralel analiz" in m.label for m in at.metric)


def test_local_fit_page():
    at = _run("Yerel Bağımsızlık ve Madde Uyumu", "lf_run")
    assert any("Ortalama Q3" in m.label for m in at.metric)


def test_local_fit_disabled_for_polytomous():
    df = pd.DataFrame(np.random.default_rng(0).integers(0, 4, (200, 5)).astype(float), columns=list("ABCDE"))
    at = _app(df, "Yerel Bağımsızlık ve Madde Uyumu"); at.run()
    assert not at.exception and any("S-X²" in e.value for e in at.error)


def test_exports_escape_item_labels_and_keep_inference_closed():
    import io,zipfile
    from openpyxl import load_workbook
    from ui.dimensionality_localfit_view import _safe,_xlsx,_zip,LOCAL_METHOD
    frame=pd.DataFrame({'=COLUMN':['=VALUE',None],'p':[float('nan'),float('nan')]},index=['=ITEM','normal'])
    safe=_safe(frame)
    assert safe.columns[0].startswith("'") and safe.index[0].startswith("'")
    book=_xlsx({'S-X2':frame});ws=load_workbook(io.BytesIO(book)).active
    assert all(c.data_type!='f' for row in ws for c in row)
    assert pd.read_excel(io.BytesIO(book)).p.isna().all()
    package=_zip({'report.xlsx':book,'report.csv':safe.to_csv().encode(),'METHOD.txt':LOCAL_METHOD.encode()})
    with zipfile.ZipFile(io.BytesIO(package)) as z:assert 'NOT CERTIFIED' in z.read('METHOD.txt').decode()
