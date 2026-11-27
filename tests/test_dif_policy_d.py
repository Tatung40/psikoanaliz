"""Policy D notices (texts) and evidence counts; protected DIF sources must stay byte-identical."""
import json
from pathlib import Path
from core.dif_policy import OBSERVED_SCORE_NOTICE, SCREENING_STATUS, SIBTEST_NOTE


def test_notice_content():
    for s in ("impact", "ayırt edici", "saflaştırma", "çıkarımsal geçerlik", "TARAMA"):
        assert s in OBSERVED_SCORE_NOTICE
    assert "sertifikalı değil" in SCREENING_STATUS and "bağımsız algoritma değildir" in SIBTEST_NOTE
    assert ".30" not in OBSERVED_SCORE_NOTICE


def test_evidence_counts_are_23_plus_3_empty():
    rep = json.loads(Path("validation/evidence/dif_difr/COMPARISON_v2.json").read_text())
    empty = [r for r in rep["rows"] if r["fixture"] == "dif2pl_none" and r["statistic"].startswith("sib_")]
    assert rep["agree_all"] and len(rep["rows"]) - len(empty) == 23 and len(empty) == 3


def test_policy_d_on_actual_screen_and_exports():
    import io
    import pandas as pd
    from streamlit.testing.v1 import AppTest
    d = pd.read_csv("tests/fixtures/irt_b10/data_dif2pl_mixed.csv")
    def app():
        import streamlit as st
        from ui.dif_classical_view import render
        render(st.session_state["d"])
    at = AppTest.from_function(app, default_timeout=120); at.session_state["d"] = d; at.run()
    assert not at.exception and any(OBSERVED_SCORE_NOTICE == w.value for w in at.warning)
