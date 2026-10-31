"""Regression: actual Python mirt call must be captured, not silently omitted."""
from pathlib import Path


def test_2pl_engine_call_is_recorded_for_python_code_panel():
    source = Path("app.py").read_text(encoding="utf-8")
    assert 'record_public_call("core.irt_python_mirt_adapter", "fit_python_mirt",' in source
    assert "record_displayed(external_fit)" in source


def test_2pl_word_error_is_visible():
    source = Path("app.py").read_text(encoding="utf-8")
    assert 'st.error(f"Akademik 2PL Word raporu hazırlanamadı:' in source
