"""Fresh project modules after an in-place redeploy.

Streamlit Cloud pulls a new commit into a running Python process. Project modules imported under the
previous commit stay in sys.modules, so new code importing a new name from them fails at start-up
(e.g. ImportError: cannot import name 'displayed' from 'core.reproducible'). Before the project imports,
every page calls purge_stale(). A module is stale when its file changed since this guard recorded it, or,
when it was imported before the guard ran (first run in a process), when it lacks a top-level name that its
current file defines. If any module is stale, all project modules are dropped and re-imported.
"""
import ast
import os
import sys

ROOTS = ("core", "ui", "reports")
_RECORD = "_psikoanaliz_module_mtimes"


def _project_modules():
    return {name: module for name, module in list(sys.modules.items())
            if name.split(".")[0] in ROOTS and getattr(module, "__file__", None)}


def _mtime(path):
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


def _defined_names(path):
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except (OSError, SyntaxError, ValueError):
        return set()
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def _stale(name, module, record):
    if name in record:
        return record[name] != _mtime(module.__file__)
    return not _defined_names(module.__file__) <= set(vars(module))


def purge_stale():
    """Drop all project modules when any loaded one is stale (see module docstring); return dropped names."""
    record = sys.__dict__.setdefault(_RECORD, {})
    loaded = _project_modules()
    if not any(_stale(name, module, record) for name, module in loaded.items()):
        return []
    for name in loaded:
        sys.modules.pop(name, None)
    record.clear()
    return sorted(loaded)


def record_loaded():
    """Remember the files of the project modules this run imported."""
    record = sys.__dict__.setdefault(_RECORD, {})
    for name, module in _project_modules().items():
        record[name] = _mtime(module.__file__)
