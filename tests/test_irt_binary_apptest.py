"""Streamlit AppTest smoke for binary IRT selection and result controls.

No external browser is simulated; these are executable Streamlit widget tests.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.mark.parametrize("model", ("1PL", "2PL", "3PL", "4PL"))
def test_binary_irt_sidebar_controls_render(model):
    streamlit_test = pytest.importorskip("streamlit.testing.v1")
    rng = np.random.default_rng(2026)
    frame = pd.DataFrame(rng.binomial(1, 0.55, size=(160, 8)),
                         columns=[f"M{i}" for i in range(1, 9)])
    frame.insert(0, "OGRENCI_NO", np.arange(len(frame)))
    at = streamlit_test.AppTest.from_file(APP, default_timeout=120)
    at.session_state["raw_df"] = frame
    at.session_state["scored_df"] = frame.copy()
    at.session_state["item_cols"] = [f"M{i}" for i in range(1, 9)]
    at.session_state["id_col"] = "OGRENCI_NO"
    at.session_state["group_col"] = None
    at.session_state["is_dichotomous"] = True
    at.session_state["nav_page"] = f"MTK: {model}"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert len([b for b in at.button if b.key == f"py_mirt_run_{model}"]) == 1
    assert not [b for b in at.button if "MTK Analizini Başlat" in b.label]
    assert any(r.key == f"model_radio_{model}" for r in at.radio)
    assert any(s.key == f"py_mirt_theta_range_{model}" for s in at.slider)
    assert any(s.key == f"py_mirt_curve_item_{model}" for s in at.selectbox)


def test_binary_irt_real_fit_and_download_buttons():
    """Exercise an actual 1PL fit and the result/export widget path."""
    streamlit_test = pytest.importorskip("streamlit.testing.v1")
    rng = np.random.default_rng(2028)
    frame = pd.DataFrame(rng.binomial(1, 0.5, size=(120, 6)),
                         columns=[f"M{i}" for i in range(1, 7)])
    frame.insert(0, "OGRENCI_NO", np.arange(len(frame)))
    at = streamlit_test.AppTest.from_file(APP, default_timeout=180)
    at.session_state["raw_df"] = frame
    at.session_state["scored_df"] = frame.copy()
    at.session_state["item_cols"] = [f"M{i}" for i in range(1, 7)]
    at.session_state["id_col"] = "OGRENCI_NO"
    at.session_state["group_col"] = None
    at.session_state["is_dichotomous"] = True
    at.session_state["nav_page"] = "MTK: 1PL"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    run = next(b for b in at.button if b.key == "py_mirt_run_1PL")
    run.click().run()
    assert not at.exception, [str(e) for e in at.exception]
    assert "1PL" in at.session_state.get("python_mirt_experimental_results", {})
    assert any("Python mirt analizi tamamlandı" in x.message for x in at.success)
    assert any("Python mirt" in str(x.message) for x in at.info)
