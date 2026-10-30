"""Start-up guards: the app and its pages open, every project from-import resolves, and modules left over
from a previous commit (Streamlit Cloud in-place redeploy) are re-imported instead of raising ImportError."""
import ast
import importlib
import sys
import types
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import project_modules

ROOT = Path(__file__).resolve().parents[1]
ENTRY = [ROOT / "app.py", *sorted((ROOT / "pages").glob("*.py"))]
SOURCES = ENTRY + sorted(p for d in ("core", "ui", "reports") for p in (ROOT / d).glob("*.py"))


@pytest.fixture
def restore_project_modules():
    """Purging re-imports project modules; give later tests their original module objects back."""
    saved = {n: m for n, m in sys.modules.items() if n.split(".")[0] in project_modules.ROOTS}
    yield
    for name in [n for n in sys.modules if n.split(".")[0] in project_modules.ROOTS]:
        del sys.modules[name]
    sys.modules.update(saved)


def test_every_project_from_import_resolves():
    missing = []
    for path in SOURCES:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 \
                    and node.module.split(".")[0] in project_modules.ROOTS:
                module = importlib.import_module(node.module)
                missing += [f"{path.name}: {node.module}.{a.name}" for a in node.names
                            if a.name != "*" and not hasattr(module, a.name)
                            and importlib.util.find_spec(f"{node.module}.{a.name}") is None]
    assert not missing


@pytest.mark.parametrize("entry", ENTRY, ids=[p.name for p in ENTRY])
def test_entry_page_opens_without_exception(entry):
    at = AppTest.from_file(str(entry), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("entry", ENTRY, ids=[p.name for p in ENTRY])
def test_stale_module_from_previous_commit_is_reimported(entry, monkeypatch, restore_project_modules):
    """Reproduces the live failure: core.reproducible loaded before `displayed` existed."""
    real = importlib.import_module("core.reproducible")
    stale = types.ModuleType("core.reproducible")
    stale.__file__ = real.__file__
    for name in ("begin", "finish", "instrument", "public_call", "bundle", "python_code", "capture_view"):
        setattr(stale, name, getattr(real, name))
    monkeypatch.setitem(sys.modules, "core.reproducible", stale)
    monkeypatch.setattr(sys, project_modules._RECORD, {}, raising=False)
    at = AppTest.from_file(str(entry), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert hasattr(sys.modules["core.reproducible"], "displayed")


def test_fresh_modules_are_kept_and_changed_files_trigger_reimport(monkeypatch, restore_project_modules):
    importlib.import_module("core.reproducible")
    monkeypatch.setattr(sys, project_modules._RECORD, {}, raising=False)
    assert project_modules.purge_stale() == []            # current modules: nothing dropped
    project_modules.record_loaded()
    record = getattr(sys, project_modules._RECORD)
    record["core.reproducible"] -= 1                       # file changed after it was recorded
    dropped = project_modules.purge_stale()
    assert "core.reproducible" in dropped and "core.reproducible" not in sys.modules
    importlib.import_module("core.reproducible")
