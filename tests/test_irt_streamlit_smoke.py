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


def test_report_exports_and_scientific_caveat():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    for name in (
        "python_mirt_{model}_parameters.csv",
        "python_mirt_{model}_parameters.xlsx",
        "python_mirt_{model}_diagnostics.csv",
        "python_mirt_{model}_all_curves.csv",
        "python_mirt_{model}_tif.csv",
    ):
        assert name in source
    assert "Parametre kurtarma ve bağımsız karşılaştırma tamamlanmadığından" in source


def test_legacy_dependency_inventory_script_parses():
    script = ROOT / "validation" / "audit_legacy_irt_imports.py"
    ast.parse(script.read_text(encoding="utf-8"))


def test_3pl_4pl_empirical_recovery_warnings_are_visible_in_ui_source():
    # Superseded 2-seed warning text replaced by the pre-registered 20-replication usage policy.
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "from core.irt_binary_usage_policy import usage_status" in source
    assert "Bilimsel kullanım sınırı: " in source
    assert 'estimation=estimation' in source
    from core.irt_binary_usage_policy import usage_status
    assert usage_status("4PL", "ML", 10000, 30)[0] == "not_supported"
    assert usage_status("4PL", "MAP", 1000, 12)[0] == "caution"
    assert usage_status("3PL", "ML", 3000, 30)[0] == "caution"


def test_polytomous_fit_passes_explicit_category_count():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert 'n_categories=(int(external_data.to_numpy().max()) + 1)' in source
    assert 'if model in ("GRM", "PCM", "GPCM", "NRM") else None' in source
