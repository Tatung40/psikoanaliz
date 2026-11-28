"""Operation-count performance contract (no wall-clock thresholds).

Unchanged inputs must not rebuild exports or figures; display-only changes rebuild only the affected artifact
and never refit; results stay in per-session state only.
"""
import sys
from pathlib import Path
import pytest

pytest.importorskip("mirt")
from streamlit.testing.v1 import AppTest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_reproducible_page_reruns as T

ROOT = Path(__file__).resolve().parents[1]
CASES = {c[0]: c for c in T.CASES}


def counting(monkeypatch, module, name, counts):
    original = getattr(module, name)
    def wrapper(*a, **k):
        counts[name] = counts.get(name, 0) + 1
        return original(*a, **k)
    monkeypatch.setattr(module, name, wrapper)


def started(page):
    _, ds, ack, button, _ = CASES[page]
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=900)
    for k, v in T._state(ds, page).items():
        at.session_state[k] = v
    at.run()
    if ack:
        at.checkbox(key=ack).check().run()
    at.button(key=button).click().run()
    assert not at.exception
    return at


def test_rasch_report_bundle_built_once_and_session_scoped(monkeypatch):
    import ui.rasch_view as RV
    counts = {}
    counting(monkeypatch, RV, "report_bundle", counts)
    at = started("MTK: Rasch")
    assert counts["report_bundle"] == 1
    cached = at.session_state["_rasch_report_bundle"]
    for _ in range(3):
        at.run()
    assert not at.exception and counts["report_bundle"] == 1
    assert at.session_state["_rasch_report_bundle"]["out"] is cached["out"]
    other = AppTest.from_file(str(ROOT / "app.py"), default_timeout=900)
    other.session_state["nav_page"] = "MTK: Rasch"
    other.run()
    assert "_rasch_report_bundle" not in other.session_state  # independent session sees nothing


def test_mirt_figure_and_exports_not_rebuilt_on_reruns_and_display_change_never_refits(monkeypatch):
    import reports.irt_multidim_report as MR
    import ui.irt_multidim_view as V
    counts = {}
    for name in ("figure", "downloads"):
        counting(monkeypatch, MR, name, counts)
        monkeypatch.setattr(V, name, getattr(MR, name))
    at = started("Çok Boyutlu MTK (MIRT)")
    saved = at.session_state[V.KEY]; result = saved["result"]
    assert counts == {"downloads": 1, "figure": 1}  # one figure shared by display and PNG export
    assert at.get("image") and saved["bundle"]["png"] is not None
    for _ in range(3):
        at.run()
    assert not at.exception and counts == {"downloads": 1, "figure": 1}
    items = at.selectbox(key="md_plot_item").options
    at.selectbox(key="md_plot_item").select(items[1]).run()  # display-only change
    assert not at.exception and counts == {"downloads": 2, "figure": 2}
    assert at.session_state[V.KEY]["result"] is result  # no refit
    assert result["metadata"]["Status"] == saved["result"]["metadata"]["Status"]
