"""Static Streamlit entrypoint smoke checks; no browser is claimed here."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_app_parses_and_has_no_retired_native_imports():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    ast.parse(source)
    forbidden = (
        "from core.irt import", "from core.irt_multistart import",
        "from ui import irt_b10_view", "from ui import irt_b11_view",
        "from ui.irt_optimization import",
    )
    for fragment in forbidden:
        assert fragment not in source


def test_sidebar_settings_and_main_results_contract():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert 'with st.sidebar.expander("⚙️ MTK model seçimi"' in source
    assert 'with st.sidebar.expander("⚙️ MTK rapor ayarları"' in source
    assert 'with st.sidebar.expander("📐 Grafik ölçeği"' in source
    assert 'with st.sidebar.expander("📈 Eğri madde seçimi"' in source
    assert 'with st.sidebar.expander("▶ MTK analizini çalıştır"' in source
    assert 'with st.expander("📦 Tüm eğri verilerini indir"' in source
    assert 'st.plotly_chart(sem_fig, use_container_width=True)' in source


def test_no_unvalidated_native_engine_path():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "run_irt_calibration(" not in source
    assert "b11.run_dif(" not in source
    assert "b10.run_global_fit(" not in source
