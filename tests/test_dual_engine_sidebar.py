"""Navigation contracts: one analysis infrastructure, no engine-choice groups."""
import ast
from pathlib import Path

def test_engine_nav_routes_and_expanders():
    text=Path("app.py").read_text(encoding="utf-8")
    ast.parse(text)
    assert 'with st.expander("🧮 KTK Analizleri"' in text
    assert 'with st.expander("📐 MTK Modelleri ve Analizler"' in text
    assert 'with st.expander("🔬 Ayrıntılı MTK Model Raporları"' not in text
    for retired in ("EserPsychometrics Yerel KTK Motoru", "Bağımsız KTK Motoru", "Python mirt Kütüphanesi — MTK",
                    '"MTK: İkili Rapor"'):
        assert retired not in text
    assert 'args=("KTK: Çift Motor Rapor",)' in text
    assert 'selected_page.startswith("MTK: Python mirt ")' in text
    assert 'for m in ("GRM","PCM","GPCM","NRM")' in text
    assert 'args=(f"MTK: Python mirt {binary_quick_model}",)' in text

def test_library_render_imports():
    from ui.irt_python_library_view import MODELS,render
    assert len(MODELS)==9
    assert callable(render)
