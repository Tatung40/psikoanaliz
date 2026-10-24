"""Transitive import reachability of project modules from the Streamlit entrypoints.

Unlike audit_legacy_irt_imports.py (top-level entrypoint imports only), this follows every
`import`/`from ... import` (including imports inside functions and page branches) starting from app.py
and pages/*.py, and lists for each project module whether it is reachable from the UI and which
tests/scripts/validation files consume it. A module is a removal candidate only if it is unreachable
from the UI AND has no consumer anywhere (tests, scripts, validation included).
Run: python validation/legacy_reachability.py  (prints JSON)
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("core", "ui", "reports")
LEGACY = ("core.irt", "ui.irt_optimization", "ui.irt_b10_view", "ui.irt_b11_view")


def module_name(path):
    return ".".join(path.relative_to(ROOT).with_suffix("").parts)


def imports_of(path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeError):
        return set()
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
            out.update(f"{node.module}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
    return out


def analyse():
    modules = {module_name(p): p for pkg in PACKAGES for p in (ROOT / pkg).rglob("*.py") if p.name != "__init__.py"}
    graph = {m: {i for i in imports_of(p) if i in modules} for m, p in modules.items()}
    roots = [ROOT / "app.py"] + sorted((ROOT / "pages").glob("*.py"))
    frontier = set().union(*({i for i in imports_of(r) if i in modules} for r in roots))
    reachable = set()
    while frontier:
        m = frontier.pop()
        if m in reachable:
            continue
        reachable.add(m); frontier |= graph[m] - reachable
    consumers = {m: [] for m in modules}
    for folder in ("tests", "scripts", "validation"):
        for p in (ROOT / folder).rglob("*.py"):
            for i in imports_of(p):
                if i in consumers:
                    consumers[i].append(str(p.relative_to(ROOT)))
    for m, deps in graph.items():
        for d in deps:
            consumers[d].append(m)
    rows = []
    for m in sorted(modules):
        rows.append({"module": m, "reachable_from_ui": m in reachable, "legacy_engine": m in LEGACY,
                     "imports_legacy": sorted(d for d in graph[m] if d in LEGACY),
                     "n_consumers": len(set(consumers[m])), "consumers_sample": sorted(set(consumers[m]))[:6]})
    ui_paths_to_legacy = sorted(r["module"] for r in rows if r["reachable_from_ui"] and (r["legacy_engine"] or r["imports_legacy"]))
    return {"modules": rows, "ui_reachable_modules_touching_legacy_engine": ui_paths_to_legacy,
            "removal_candidates": [r["module"] for r in rows if not r["reachable_from_ui"] and r["n_consumers"] == 0]}


if __name__ == "__main__":
    print(json.dumps(analyse(), ensure_ascii=False, indent=1))
