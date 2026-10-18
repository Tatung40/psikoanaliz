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
from core.irt_simulation import simulate_irt
from ui.irt_optimization import REFINE_LABEL

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _data():
    truth = pd.DataFrame({"Madde": [f"Item_{i+1}" for i in range(5)], "a": np.linspace(1.2, 2., 5),
                          "b": np.linspace(-1., 1., 5), "c": np.linspace(.12, .2, 5), "d": 1.})
    return simulate_irt(600, truth, seed=12)[0]


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


def test_3pl_refinement_selected_runs_and_is_stored():
    at = _app("MTK: 3PL")
    assert not at.exception
    boxes = _refine_boxes(at)
    assert len(boxes) == 1 and boxes[0].value is False          # default off
    boxes[0].check().run()
    next(b for b in at.button if "MTK Analizini Başlat" in b.label).click().run()
    assert not at.exception
    assert any("EM + L-BFGS-B" in s.value for s in at.success)
    stored = at.session_state["irt_model_results"]["3PL"]["result"]
    assert stored["settings"]["refine"] is True and stored["refinement"]["requested"] is True
    assert set(stored["multistart_audit"]["İyileştirme"]) <= {"kabul edildi", "reddedildi"}
    shown = " ".join(c.value for c in at.caption)
    assert "Yöntem: EM + L-BFGS-B" in shown and "Gauss–Hermite düğümü: 31" in shown
    assert any("Kestirim yöntemi" in str(df.value.to_dict()) for df in at.dataframe)
    # toggling the option afterwards does not silently relabel the stored result
    _refine_boxes(at)[0].uncheck().run()
    assert any("farklı ayarlarla üretildi" in i.value for i in at.info)
    assert at.session_state["irt_model_results"]["3PL"]["result"]["settings"]["refine"] is True
