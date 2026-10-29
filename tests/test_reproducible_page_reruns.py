"""The shared Python code panel keeps the actual calculation after ordinary reruns on every screen that
shows a cached result, and links results reused from another page (model comparison)."""
from pathlib import Path

import pandas as pd
import pytest

from core import example_datasets as E
from core.reproducible import begin, displayed, finish, recorded

pytest.importorskip("mirt")
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
CASES = [  # page, example dataset, consent checkbox, run button, expected recorded producer
    ("MTK: Rasch", "irt_000", None, "rasch_run", "rasch_analysis"),
    ("MTK: Python mirt DIF", "dif_075", None, "pmd_run", "analyze"),
    ("MTK: Gözlenen Puan DIF", "dif_075", None, "cdif_run", "dif_screen"),
    ("MTK: Çok Modelli DIF Taraması", "dif_075", None, "dif_multi_start", "screen_mtk_groups"),
    ("MTK: GRM Formel DIF", "dif_081", "grm_ack", "grm_run", "test_dif"),
    ("MTK: Çok Kategorili Formel DIF", "dif_081", "poly_ack", "poly_run", "test_dif"),
    ("MTK: Varsayım Denetimi", "irt_006", None, "irt_check_run", "preflight"),
    ("Boyutluluk ve Faktör Analizi", "irt_006", None, "dim_run", "dimensionality_analysis"),
    ("Yerel Bağımsızlık ve Madde Uyumu", "irt_006", None, "lf_run", "run"),
    ("Çok Boyutlu MTK (MIRT)", "mirt_033", None, "md_run", "fit"),
    ("MTK: 1PL", "irt_003", None, "py_mirt_run_1PL", "fit_python_mirt"),
    ("MTK: 4PL", "irt_012", None, "py_mirt_run_4PL", "fit_python_mirt"),
]


def _state(dataset, page):
    b = E.materialize(dataset)
    state = dict(raw_df=E.input_frame(b), item_cols=b["items"], id_col=b["id"], group_col=b["group"],
                 answer_key=b["key"], scored_df=E.scored(b) if b["key"] else b["data"].copy(), nav_page=page)
    observed = set(pd.Series(b["data"][b["items"]].to_numpy().ravel()).dropna().unique())
    state["is_dichotomous"] = bool(b["key"]) or bool(observed) and observed <= {0, 1}
    state.update(b["settings"])
    if b["spec"].model in ("GRM", "PCM", "GPCM", "NRM"):   # as ui.example_datasets_view.activate
        state.update(b14_model=b["spec"].model, b14_categories="A,B,C,D", grm_categories=4, poly_categories=4,
                     poly_labels='["A","B","C","D"]')
    return state


def _manifest(at):
    return at.session_state["_repro_history"].manifest()


@pytest.mark.parametrize("page,dataset,ack,button,producer", CASES, ids=[c[0] for c in CASES])
def test_code_panel_keeps_the_calculation_after_reruns(page, dataset, ack, button, producer):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    for key, value in _state(dataset, page).items():
        at.session_state[key] = value
    at.run()
    if ack:
        at.checkbox(key=ack).check().run()
    at.button(key=button).click().run()
    first = [c["function"] for c in _manifest(at)["calls"]]
    assert producer in first
    for _ in range(2):                       # widget change / tab switch reruns use the cached result
        at.run()
        manifest = _manifest(at)
        assert not at.exception
        assert manifest["status"] == "GENERATED / UNVERIFIED", manifest["unsupported"]
        assert producer in [c["function"] for c in manifest["calls"]]
        assert not any("UNSUPPORTED" in i.value for i in at.info)


def test_bootstrap_dif_stays_explicitly_unsupported_after_rerun():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    for key, value in _state("dif_075", "MTK: 3PL/4PL Bootstrap DIF").items():
        at.session_state[key] = value
    at.run(); at.button(key="b34_run").click().run(); at.run()
    manifest = _manifest(at)
    assert manifest["status"] == "UNSUPPORTED"
    assert any("bootstrap DIF inference is gated" in u for u in manifest["unsupported"])


def test_comparison_page_links_fits_from_the_model_pages_and_closes_capture():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    for key, value in _state("irt_006", "MTK: Karşılaştırma").items():
        at.session_state[key] = value
    at.run()
    for model in ("1PL", "2PL"):
        at.session_state["nav_page"] = f"MTK: {model}"
        at.run(); at.button(key=f"py_mirt_run_{model}").click().run()
    at.session_state["nav_page"] = "MTK: Karşılaştırma"
    at.run()
    recorder = at.session_state["_repro_history"]
    assert recorder.page == "MTK: Karşılaştırma"             # st.stop() no longer skips the panel
    calls = recorder.manifest()["calls"]
    models = [dict(c["arguments"]["values"])["model"] for c in calls if c["function"] == "fit_python_mirt"]
    assert sorted(models) == ["1PL", "2PL"]
    assert [c["function"] for c in calls].count("library_fit_diagnostics") == 2


def test_missing_data_page_keeps_applied_imputation_after_rerun():
    at = AppTest.from_file(str(ROOT / "pages" / "90_Kayip_Veri_Analizi_ve_Atama.py"), default_timeout=600)
    for key, value in _state("missing_112", "Eksik Veri ve Atama").items():
        at.session_state[key] = value
    at.run()
    next(b for b in at.button if "uygula" in (b.label or "")).click().run()
    at.run()
    functions = [c["function"] for c in _manifest(at)["calls"]]
    assert "apply" in functions and _manifest(at)["status"] == "GENERATED / UNVERIFIED"


def test_displayed_links_only_unchanged_values_of_the_same_or_other_recorded_pages():
    @recorded
    def produce(x):
        return x * 2
    first = begin("page")
    kept = produce(pd.DataFrame({"a": [1., 2.]}))
    edited = produce(pd.DataFrame({"a": [3., 4.]}))
    finish()
    edited.iloc[0, 0] = 99.                  # changed in place after recording: never attributed
    second = begin("page", previous=first)
    displayed(kept, edited, pd.DataFrame({"a": [5.]}))
    finish()
    assert len(second.calls) == 1 and second.calls[0]["function"] == "produce"
    other = begin("other page", others=(first,))
    displayed(kept)
    finish()
    assert [c["function"] for c in other.calls] == ["produce"]
