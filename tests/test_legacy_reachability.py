"""Guard: only the explicitly labeled B13 research page may reach the retired native 1PL-4PL engine."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reach", ROOT / "validation" / "legacy_reachability.py")
reach = importlib.util.module_from_spec(spec); spec.loader.exec_module(reach)

ALLOWED = {"ui.irt_b13_view", "core.irt_b13_curves", "core.irt", "core.irt_hessian", "core.irt_itemfit",
           "core.irt_stationarity"}


def test_ui_reachable_legacy_paths_are_confined_to_b13_research_page():
    result = reach.analyse()
    assert set(result["ui_reachable_modules_touching_legacy_engine"]) <= ALLOWED
    rows = {r["module"]: r for r in result["modules"]}
    for active in ("core.irt_python_mirt_adapter", "core.irt_rsm", "core.dif_classical", "ui.irt_rsm_view",
                   "ui.dif_classical_view", "core.irt_b14_persons_info"):
        assert rows[active]["reachable_from_ui"] and not rows[active]["imports_legacy"]


def test_b13_page_discloses_retired_engine():
    assert "emekliye ayrılmış yerel" in (ROOT / "ui" / "irt_b13_view.py").read_text(encoding="utf-8")
