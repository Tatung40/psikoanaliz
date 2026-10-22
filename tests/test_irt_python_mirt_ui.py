"""Guard experimental external engine against contaminating native inference."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

APP=str(Path(__file__).resolve().parents[1]/"app.py")

@pytest.mark.parametrize("model",["1PL","2PL","3PL","4PL"])
def test_external_engine_is_opt_in_and_does_not_overwrite_native(model):
    st=pytest.importorskip("streamlit.testing.v1")
    from core.irt_python_mirt_adapter import fit_python_mirt
    from scipy.special import expit
    rng=np.random.default_rng(123)
    theta=rng.normal(size=150)
    b=np.linspace(-1,1,8)
    x=(rng.random((150,8))<expit(theta[:,None]-b)).astype(int)
    d=pd.DataFrame(x,columns=[f"M{i}" for i in range(8)])
    at=st.AppTest.from_file(APP,default_timeout=120)
    at.session_state["raw_df"]=d
    at.session_state["scored_df"]=d.copy()
    at.session_state["item_cols"]=list(d.columns)
    at.session_state["id_col"]=None
    at.session_state["group_col"]=None
    at.session_state["is_dichotomous"]=True
    at.session_state["nav_page"]=f"MTK: {model}"
    at.run()
    assert not at.exception
    assert f"py_mirt_run_{model}" in {b.key for b in at.button}
    assert not at.session_state.get("python_mirt_experimental_results",{})
    at.button(key=f"py_mirt_run_{model}").click().run()
    assert not at.exception
    assert at.session_state["python_mirt_experimental_results"][model]["fit"].engine=="python_mirt"
    assert not at.session_state.get("irt_model_results",{})
