"""Navigation contracts: users can select native or Python engines separately."""
import ast
from pathlib import Path

def test_engine_nav_routes_and_expanders():
    text=Path("app.py").read_text(encoding="utf-8")
    ast.parse(text)
    assert 'with st.expander("🧮 EserPsychometrics Yerel KTK Motoru"' in text
    assert 'with st.expander("🐍 Python Kütüphaneleri / Bağımsız KTK Motoru"' in text
    assert 'with st.expander("🧮 EserPsychometrics Yerel MTK Motoru"' in text
    assert 'with st.expander("🐍 Python mirt Kütüphanesi — MTK"' in text
    assert 'selected_page.startswith("MTK: Python mirt ")' in text
    assert '("1PL","2PL","3PL","4PL","GRM","PCM","GPCM","NRM")' in text

def test_library_render_imports():
    from ui.irt_python_library_view import MODELS,render
    assert len(MODELS)==8
    assert callable(render)
