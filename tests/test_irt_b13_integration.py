"""B13 integration smoke: real Streamlit route and scientific native tables."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from ui.irt_b13_view import native_fit,prepare

def sample():
    rng=np.random.default_rng(13)
    z=rng.normal(size=160)
    b=np.linspace(-1,1,8)
    return pd.DataFrame((rng.random((160,8))<expit(z[:,None]-b)).astype(int),
                        columns=[f"M{i}" for i in range(8)])

@pytest.mark.parametrize("model",["Rasch","1PL","2PL"])
def test_b13_native_tables(model):
    r=native_fit(sample(),model)
    assert len(r["items"])==8
    assert len(r["persons"])==160
    assert r["checks"]["passed"]
    assert r["overview"].iloc[0]["Model"]==model

def test_b13_rejects_bad_matrix():
    d=sample().astype(float)
    d.iloc[0,0]=np.nan
    with pytest.raises(ValueError,match="Eksiksiz"):
        prepare(d,"2PL")

def test_b13_streamlit_screen():
    st=pytest.importorskip("streamlit.testing.v1")
    at=st.AppTest.from_file(str(Path(__file__).resolve().parents[1]/"app.py"),default_timeout=120)
    d=sample()
    for k,v in {"raw_df":d,"scored_df":d.copy(),"item_cols":list(d.columns),
                "id_col":None,"group_col":None,"is_dichotomous":True,
                "nav_page":"MTK: İkili Rapor"}.items():
        at.session_state[k]=v
    at.run()
    assert not at.exception
    assert any(b.key=="b13_run" for b in at.button)
    at.button(key="b13_run").click().run()
    assert not at.exception
    assert "b13_latest" in at.session_state
