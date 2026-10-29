"""B7: real Streamlit interaction test of the IRT page (streamlit.testing AppTest).

Drives app.py headlessly: data placed in session_state, MTK page selected, the
refinement checkbox toggled and the analysis button clicked.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest
# Label of the removed native refinement option; it must never reappear on library routes.
REFINE_LABEL = "Gelişmiş kestirim (EM + L-BFGS-B)"

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _data():
    truth = pd.DataFrame({"Madde": [f"Item_{i+1}" for i in range(5)], "a": np.linspace(1.2, 2., 5),
                          "b": np.linspace(-1., 1., 5), "c": np.linspace(.12, .2, 5), "d": 1.})
    rng = np.random.default_rng(12)
    theta = rng.standard_normal(600)[:, None]
    a, b, c, d = (truth[k].to_numpy(float)[None, :] for k in ("a", "b", "c", "d"))
    p = c + (d - c) / (1 + np.exp(-a * (theta - b)))
    return pd.DataFrame((rng.random(p.shape) < p).astype(int), columns=truth["Madde"])


def _app(page):
    at = AppTest.from_file(APP, default_timeout=120)
    df = _data()
    at.session_state["raw_df"] = df
    at.session_state["scored_df"] = df.copy()
    at.session_state["item_cols"] = list(df.columns)
    at.session_state["is_dichotomous"] = True
    at.session_state["nav_page"] = page
    return at.run()


def _refine_boxes(at):
    return [c for c in at.checkbox if c.label == REFINE_LABEL]


@pytest.mark.parametrize("page", ["MTK: 1PL", "MTK: 2PL"])
def test_option_hidden_for_1pl_2pl(page):
    at = _app(page)
    assert not at.exception
    assert _refine_boxes(at) == []


def test_3pl_library_controls_and_report_toggles_do_not_refit():
    """Retired native refinement must stay absent from the library-only route."""
    at = _app("MTK: 3PL")
    assert not at.exception and _refine_boxes(at) == []
    assert not [b for b in at.button if "MTK Analizini Başlat" in b.label]
    at.button(key="py_mirt_run_3PL").click().run()
    assert not at.exception
    stored = at.session_state["python_mirt_experimental_results"]["3PL"]["fit"]
    assert stored.model == "3PL" and stored.engine == "python_mirt"
    assert stored.n_items == 5
    at.checkbox(key="py_mirt_show_diagnostics_3PL").uncheck().run()
    assert not at.exception
    assert at.session_state["python_mirt_experimental_results"]["3PL"]["fit"] is stored
    assert "3PL" not in at.session_state.get("irt_model_results", {})
