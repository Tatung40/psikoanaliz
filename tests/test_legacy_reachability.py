"""Guard: the native ("Eser") binary IRT engine is removed; nothing may import it again."""
import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reach", ROOT / "validation" / "legacy_reachability.py")
reach = importlib.util.module_from_spec(spec); spec.loader.exec_module(reach)

REMOVED = (
    "core.irt",
    "core.irt_hessian",
    "core.irt_stationarity",
    "core.irt_b13_curves",
    "core.irt_b13_diagnostics",
    "core.irt_b13_compare",
    "reports.irt_b13_png",
    "ui.irt_b13_view",
    "core.irt_b12_compare",
    "core.irt_b12_audit",
    "core.irt_b12_reference",
    "core.irt_binary_unified",
    "core.irt_missing_em",
    "core.irt_missing_likelihood",
    "core.irt_model_reliability",
    "core.irt_multistart",
    "core.irt_multistart_validation",
    "core.irt_quadrature_validation",
    "core.irt_se_coverage",
    "core.irt_simulation",
    "core.irt_simulation_reporting",
    "core.irt_mixed",
    "core.irt_mixed_se",
    "core.irt_monte_carlo_precision",
    "core.irt_truth_scenarios",
    "core.irt_engine_policy",
    "core.irt_engine_registry",
    "core.irt_globalfit",
    "core.irt_dif34",
    "ui.irt_b10_view",
    "ui.irt_b11_view",
    "ui.irt_optimization",
)


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
            yield from (f"{node.module}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Import):
            yield from (a.name for a in node.names)


def test_removed_engine_modules_are_gone_and_never_imported():
    for name in REMOVED:
        assert not (ROOT / (name.replace(".", "/") + ".py")).exists(), name
    offenders = []
    for path in ROOT.rglob("*.py"):
        if any(part in (".git", "__pycache__") for part in path.parts):
            continue
        offenders += [f"{path.relative_to(ROOT)}: {m}" for m in _imports(path) if m in REMOVED]
    assert not offenders, offenders


def test_active_library_paths_are_ui_reachable_without_legacy_imports():
    result = reach.analyse()
    assert result["ui_reachable_modules_touching_legacy_engine"] == []
    rows = {r["module"]: r for r in result["modules"]}
    for active in ("core.irt_python_mirt_adapter", "core.irt_rsm", "core.dif_classical", "ui.irt_rsm_view",
                   "ui.dif_classical_view", "core.irt_b14_persons_info", "core.local_fit", "core.irt_itemfit",
                   "core.irt_rasch_wright", "ui.wright_map", "core.irt_person_abilities"):
        assert rows[active]["reachable_from_ui"] and not rows[active]["imports_legacy"]
