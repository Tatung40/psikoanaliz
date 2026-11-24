"""One calculation click on each IRT engine yields all outputs and a replayable recipe (no second click/rerun).

Shared components under test: core.reproducible (late-import instrumentation), ui.academic_word (display-only
widgets), ui.irt_python_library_view (extras/bundle/Wright with the click), ui.irt_rsm_view (no cache copies),
ui.validation_details (one collapsed technical section).
"""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mirt")
from streamlit.testing.v1 import AppTest

from tests.test_reproducible_page_reruns import _state
from tests.test_reproducible_result_fields import _replay
from ui.reproducible_capture import python_code, replay_manifest
from ui.validation_details import INTERNAL, TITLE

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _binary(model):
    return lambda s: s["python_mirt_experimental_results"][model]["fit"].fit.log_likelihood


def _library(model):
    return lambda s: s[f"python_mirt_fit_{model}"]["outcome"].fit.log_likelihood


CASES = [  # page, dataset, run button, display-only widget, producer, stored log-likelihood, downstream call
    *[(f"MTK: {m}", d, f"py_mirt_run_{m}", f"py_mirt_curve_item_{m}", "fit_python_mirt", _binary(m),
       "python_mirt_binary_parameters") for m, d in (("1PL", "irt_003"), ("2PL", "irt_006"), ("3PL", "irt_009"),
                                                     ("4PL", "irt_012"))],
    *[(f"MTK: Python mirt {m}", d, f"python_mirt_start_{m}", None, "fit_python_mirt", _library(m), "build_report")
      for m, d in (("Rasch", "irt_000"), ("GRM", "irt_015"), ("PCM", "irt_018"), ("GPCM", "irt_021"),
                   ("NRM", "irt_024"))],
    ("MTK: Python mirt 2PL", "irt_006", "python_mirt_start_2PL", "python_mirt_extra_kind_2PL", "fit_python_mirt",
     _library("2PL"), "parameter_inference"),
    ("MTK: RSM", "irt_027", "rsm_run", "rsm_curve_item", "fit",
     lambda s: s["rsm_request"]["result"]["fit"]["loglik"], "category_curves"),
]


def _loglik(result):
    fit = getattr(result, "fit", result)
    return fit["loglik"] if isinstance(fit, dict) else fit.log_likelihood


def _check_recipe(at, page, producer, stored, downstream):
    manifest = replay_manifest(at.session_state["_repro_pages"][page])
    functions = [c["function"] for c in manifest["calls"]]
    assert manifest["unsupported"] == [] and manifest["status"] == "GENERATED / UNVERIFIED", manifest["unsupported"]
    assert functions.count(producer) == 1 and downstream in functions, functions   # one fit, full downstream
    compile(python_code(manifest), "analysis.py", "exec")
    replayed = _replay(manifest)
    assert _loglik(replayed[functions.index(producer)]) == pytest.approx(stored(at.session_state), abs=1e-8)


@pytest.mark.parametrize("page,dataset,button,view,producer,stored,downstream", CASES, ids=[c[0] for c in CASES])
def test_single_click_yields_complete_outputs_and_replayable_code(page, dataset, button, view, producer, stored,
                                                                  downstream):
    at = AppTest.from_file(APP, default_timeout=600)
    for key, value in _state(dataset, page).items():
        at.session_state[key] = value
    at.run()
    at.button(key=button).click().run()                       # the only click; no extra rerun
    assert not at.exception, [str(e.value) for e in at.exception]
    _check_recipe(at, page, producer, stored, downstream)
    downloads = {b.key for b in at.get("download_button")}
    assert {"repro_py", "repro_zip"} <= downloads
    assert not [w for w in at.warning if "eski Word ve Python kodu kapatıldı" in str(w.value)]
    # Routine notices sit in one collapsed section; visible text exposes no internal identifiers.
    sections = [e for e in at.expander if e.label == TITLE]
    assert len(sections) <= 1 and all(not getattr(e.proto, "expanded", False) for e in sections)
    inside = [str(x.value) for e in sections for x in list(e.info) + list(e.success) + list(e.caption)]
    for element in list(at.info) + list(at.success) + list(at.caption):
        if str(element.value) in inside:
            inside.remove(str(element.value))
        elif not str(element.value).startswith(("GENERATED", "Bağımsız çalıştırma")):
            assert not INTERNAL.search(str(element.value)), element.value
    if view:                                                  # display-only change: no refit, code stays valid
        widget = next(w for w in at.selectbox if w.key == view)
        widget.select(widget.options[-1]).run()
        assert not at.exception
        _check_recipe(at, page, producer, stored, downstream)
        assert downloads <= {b.key for b in at.get("download_button")}


def test_risks_and_failures_stay_visible():
    import streamlit as st
    from ui.validation_details import routine
    assert routine("info", "Aşağıdaki özet, Python mirt kütüphanesinin doğrudan ürettiği çıktıdır.")
    assert routine("success", "Python mirt analizi tamamlandı.")
    for text in ("Veri veya analiz ayarları değişti; yeniden başlatın.", "Kestirim yakınsamadı.",
                 "Kayıp veri işlemi uygulandı.", "Bilimsel kullanım sınırı: yorumlanmamalı."):
        assert not routine("info", text)
    assert routine("caption", "Motor: python_mirt | Model: 2PL") and routine("caption", "ValueError: x")
    assert routine("caption", "core.irt_python_mirt_adapter.fit_python_mirt: no faithful public serialization")
    assert not routine("caption", "Ön kontrol: 300 kişi × 8 madde; eksik yanıt: 0.")
    assert st.warning.__name__ == "warning" and not getattr(st.warning, "__validation_details__", False)
