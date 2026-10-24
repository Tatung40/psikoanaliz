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
        'st.warning("Motor politikası:', 1)[0]
    for forbidden in ("Gauss-Hermite düğüm sayısı", "Azami EM yinelemesi",
                      "a üst sınırı", "d alt sınırı", "Bağımsız başlangıç sayısı"):
        assert forbidden not in block
    assert "Etkin kestirim motoru: Python mirt" in block


def test_retired_native_calibration_button_and_call_are_removed():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "MTK Analizini Başlat — eski yerel motor devre dışı" not in source
    assert "result=run_irt_calibration(data,current_settings)" not in source
    assert 'st.session_state.pop("irt_model_results", None)' in source


def test_native_results_ui_is_not_present():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert 'saved = st.session_state.get("irt_model_results"' not in source
    assert 'r = saved.get("result")' not in source
    assert 'b11.run_dif(' not in source
    assert 'b10.run_global_fit(' not in source
