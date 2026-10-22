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


def test_apptest_4pl_diagnostics_and_result_isolation():
    """Real 4PL fit -> global fit/Q3 -> rerun: no stale diagnostics."""
    st = pytest.importorskip("streamlit.testing.v1")
    df = pd.read_csv(FIXTURES / "data_irt4pl_ok.csv").iloc[:350].copy()
    df.insert(0, "OGRENCI_NO", np.arange(len(df)))
    at = st.AppTest.from_file(APP, default_timeout=600)
    at.session_state["raw_df"] = df
    at.session_state["scored_df"] = df.copy()
    at.session_state["item_cols"] = ITEMS
    at.session_state["id_col"] = "OGRENCI_NO"
    at.session_state["group_col"] = None
    at.session_state["is_dichotomous"] = True
    at.session_state["nav_page"] = "MTK: 4PL"
    at.run()
    assert not at.exception
    next(b for b in at.button if "MTK Analizini Başlat" in b.label).click().run()
    assert not at.exception
    fit = at.session_state["irt_model_results"]["4PL"]["result"]
    assert fit["items"]["Madde"].tolist() == ITEMS
    assert {"b11_gf_run_4PL", "b11_q3_run_4PL"}.issubset(
        {b.key for b in at.button})
    at.button(key="b11_gf_run_4PL").click().run()
    assert not at.exception
    gf = at.session_state["irt_b11"][("gf", "4PL")]["result"]
    assert gf["model"] == "4PL"
    at.button(key="b11_q3_run_4PL").click().run()
    assert not at.exception
    q3 = at.session_state["irt_b11"][("q3", "4PL")]["result"]
    assert q3["model"] == "4PL"
    assert not [b for b in at.button if b.key and b.key.startswith("b10_")]
    next(b for b in at.button if "MTK Analizini Başlat" in b.label).click().run()
    assert not at.exception
    # Cache entries may be retained internally, but must not be displayed as current.
    assert not any("Model (ICC)" in str(m.value) for m in at.markdown)
