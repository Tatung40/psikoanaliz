"""Repository audit for retired IRT engine imports.

Only the Streamlit entrypoint is enforced; auxiliary legacy research modules
remain inventoried until their consumers are migrated and verified.
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY_PREFIXES = (
    "core.irt", "ui.irt_b10_view", "ui.irt_b11_view",
    "ui.irt_optimization",
)


def audit():
    inventory = []
    for folder in ("core", "ui", "pages", "reports"):
        for file in sorted((ROOT / folder).rglob("*.py")):
            try:
                tree = ast.parse(file.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeError) as exc:
                inventory.append({"file": str(file.relative_to(ROOT)),
                                  "error": type(exc).__name__})
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                else:
                    continue
                for module in modules:
                    if any(module == p or module.startswith(p + ".")
                           for p in LEGACY_PREFIXES):
                        inventory.append({"file": str(file.relative_to(ROOT)),
                                          "line": node.lineno, "import": module})
    entrypoint = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    active = []
    for node in ast.walk(entrypoint):
        if isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        elif isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        else:
            continue
        active.extend(n for n in names if any(
            n == p or n.startswith(p + ".") for p in LEGACY_PREFIXES))
    return {"legacy_import_inventory": inventory, "active_entrypoint_legacy_imports": active,
            "safe_to_remove_all_legacy_modules": False}


if __name__ == "__main__":
    report = audit()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["active_entrypoint_legacy_imports"]:
        raise SystemExit(1)
