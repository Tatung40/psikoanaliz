"""Regression guards for 2PL report visibility and reproducible code diagnostics."""
from pathlib import Path


def test_word_export_unexpected_errors_are_visible():
    app = Path("app.py").read_text(encoding="utf-8")
    assert 'except Exception as exc:\n                        st.error(f"Akademik 2PL Word raporu hazırlanamadı:' in app
    assert 'st.markdown("### 📄 Akademik Word raporu")' in app
    assert app.index('st.markdown("### 📄 Akademik Word raporu")') > app.index('if model in ("1PL", "2PL", "3PL", "4PL")')


def test_python_panel_reports_missing_or_unsupported_capture():
    view = Path("ui/reproducible_view.py").read_text(encoding="utf-8")
    assert "kaydedilmiş bir hesaplama çağrısı bulunamadı" in view
    assert "manifest.get('unsupported', [])" in view
