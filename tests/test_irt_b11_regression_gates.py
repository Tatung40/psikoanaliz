"""B11.1 regression gates: 4PL interface and safe optional B11 loading."""
import builtins
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
FIXTURES = Path(__file__).parent / "fixtures" / "irt_b11"
ITEMS = [f"M{i}" for i in range(1, 13)]


def test_b11_import_failure_does_not_crash_ctt(monkeypatch):
    """Simulate the original B11 ImportError and require the main app to render."""
    st = pytest.importorskip("streamlit.testing.v1")
    original_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ui" and "irt_b11_view" in (fromlist or ()):
            raise ImportError("B11.1 simulated missing B11 dependency")
        return original_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", blocked_import)
        at = st.AppTest.from_file(APP, default_timeout=120)
        at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert any("Psiko" in str(m.value) or "Analiz" in str(m.value)
               for m in at.markdown) or len(at.button) > 0


def test_apptest_4pl_library_first_and_no_native_results():
    """The retired native fit cannot be invoked; library route is visible.

    This is a UI safety regression, NOT a replacement for B11 diagnostics
    validation, which must be re-established for library-backed results.
    """
    st = pytest.importorskip("streamlit.testing.v1")
    df = pd.read_csv(FIXTURES / "data_irt4pl_ok.csv").iloc[:350].copy()
    df.insert(0, "OGRENCI_NO", np.arange(len(df)))
    at = st.AppTest.from_file(APP, default_timeout=120)
    at.session_state["raw_df"] = df
    at.session_state["scored_df"] = df.copy()
    at.session_state["item_cols"] = ITEMS
    at.session_state["id_col"] = "OGRENCI_NO"
    at.session_state["group_col"] = None
    at.session_state["is_dichotomous"] = True
    at.session_state["nav_page"] = "MTK: 4PL"
    at.run()
    assert not at.exception
    library_buttons = [b for b in at.button if b.key == "py_mirt_run_4PL"]
    assert len(library_buttons) == 1
    assert not library_buttons[0].disabled
    legacy_buttons = [b for b in at.button if "MTK Analizini Başlat" in b.label]
    assert len(legacy_buttons) == 1
    assert legacy_buttons[0].disabled
    assert not at.session_state.get("irt_model_results", {}).get("4PL")
