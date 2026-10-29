"""Contract tests: Python mirt settings are not silently routed to native EM.

UI may offer only controls supported by the external library adapter.
"""
from pathlib import Path
from core.irt_python_mirt_adapter import fit_python_mirt


def test_adapter_does_not_accept_legacy_optimizer_controls():
    import inspect
    parameters = inspect.signature(fit_python_mirt).parameters
    assert "qpoints" not in parameters
    assert "a_upper" not in parameters
    assert "starts" not in parameters
    assert "refine" not in parameters


def test_visible_mtk_ui_has_no_legacy_tuning_sliders():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    block = source.split('with st.sidebar.expander("⚙️ MTK model seçimi"', 1)[1].split(
        'st.warning("Etkin iki kategorili MTK motoru:', 1)[0]
    for forbidden in ("Gauss-Hermite düğüm sayısı", "Azami EM yinelemesi",
                      "a üst sınırı", "d alt sınırı", "Bağımsız başlangıç sayısı"):
        assert forbidden not in block
    assert "Etkin kestirim motoru: Python mirt" in block


def test_retired_native_calibration_button_and_call_are_removed():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "MTK Analizini Başlat — eski yerel motor devre dışı" not in source
    assert "result=run_irt_calibration(data,current_settings)" not in source
    # Retired native session keys are neither written, read nor cleaned up any more.
    for key in ('"irt_model_results"', '"irt_b10"', '"irt_b11"'):
        assert key not in source


def test_native_results_ui_is_not_present():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert 'saved = st.session_state.get("irt_model_results"' not in source
    assert '\n        r = saved.get("result")' not in source
    assert 'b11.run_dif(' not in source
    assert 'b10.run_global_fit(' not in source


def test_library_curve_and_diagnostic_download_controls():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    for marker in (
        "python_mirt_{model}_parameters.xlsx",
        "python_mirt_{model}_diagnostics.csv",
        "python_mirt_{model}_{key}_{selected_item}.csv",
        "python_mirt_{model}_tif.csv",
    ):
        assert marker in source


def test_curve_grid_settings_are_sidebar_only():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert 'with st.sidebar.expander("📐 Grafik ölçeği"' in source
    assert 'key=f"py_mirt_theta_range_{model}"' in source
    assert 'key=f"py_mirt_theta_points_{model}"' in source
    assert "np.linspace(theta_min, theta_max, theta_count)" in source


def test_all_item_curve_export_and_conditional_sem_are_present():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert 'python_mirt_{model}_all_curves.csv' in source
    assert 'Koşullu Ölçmenin Standart Hatası' in source
    assert 'conditional_sem' in source
    assert 'with st.expander("📦 Tüm eğri verilerini indir"' in source
    assert 'with st.sidebar.expander("📦 Tüm eğri verileri"' not in source


def test_plot_controls_exist_outside_fit_result_block():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    sidebar = source.index('with st.sidebar.expander("📐 Grafik ölçeği"')
    runner = source.index('with st.sidebar.expander("▶ MTK analizini çalıştır"')
    assert sidebar < runner
    assert source.index('with st.sidebar.expander("📈 Eğri madde seçimi"') < runner
    assert source.count('key=f"py_mirt_curve_item_{model}"') == 1
