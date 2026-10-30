"""Navigation regression: direct Rasch/1PL-4PL buttons open Python mirt pages."""
from pathlib import Path
import pytest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
MODELS = ("Rasch", "1PL", "2PL", "3PL", "4PL")


@pytest.mark.parametrize("model", MODELS)
def test_binary_direct_navigation(model):
    st = pytest.importorskip("streamlit.testing.v1")
    at = st.AppTest.from_file(APP, default_timeout=90).run()
    assert not at.exception
    assert not any(x.key == "binary_irt_quick_model" for x in at.selectbox)
    assert not any(x.key == "binary_irt_quick_go" for x in at.button)
    at.button(key=f"irt_nav_{model}").click().run()
    assert not at.exception
    assert at.session_state["nav_page"] == f"MTK: Python mirt {model}"
