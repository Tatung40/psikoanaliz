"""Navigation regression: Rasch/1PL-4PL quick picker opens detailed Python mirt pages."""
from pathlib import Path
import pytest

APP=str(Path(__file__).resolve().parents[1]/"app.py")
MODELS=("Rasch","1PL","2PL","3PL","4PL")

@pytest.mark.parametrize("model",MODELS)
def test_binary_dropdown_navigation(model):
    st=pytest.importorskip("streamlit.testing.v1")
    at=st.AppTest.from_file(APP,default_timeout=90).run()
    assert not at.exception
    picker=at.selectbox(key="binary_irt_quick_model")
    picker.set_value(model).run()
    assert not at.exception
    at.button(key="binary_irt_quick_go").click().run()
    assert not at.exception
    assert at.session_state["nav_page"]==f"MTK: Python mirt {model}"
