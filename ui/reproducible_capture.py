"""Reporting-only extension of the existing recorder; estimation sources stay frozen."""
import inspect
import sys
import hashlib
import io
import zipfile
from pathlib import Path
from core import reproducible as base

# Only existing public computations, never export/UI/RAG functions indiscriminately.
REPORT_COMPUTATIONS={
    'reports.irt_b14_report':{'category_frequencies','quality_report','summary_metadata','report_tables'},
    'reports.irt_b14_narrative':{'guided_report'},
    'reports.irt_b14_audit':{'interpretation_audit'},
}


def instrument(namespace=None):
    base.instrument(namespace)
    charts=sys.modules.get('ui.charts')
    if charts is not None:
        for name,function in list(vars(charts).items()):
            if name.startswith('plot_') and inspect.isfunction(function) and function.__module__=='ui.charts':
                wrapped=base.recorded(function);setattr(charts,name,wrapped)
                if namespace is not None and name in namespace:namespace[name]=wrapped
    for module_name,module in list(sys.modules.items()):
        if module is None or not module_name.startswith('reports.'):continue
        for name,function in list(vars(module).items()):
            if not inspect.isfunction(function):continue
            core_alias=(function.__module__.startswith('core.') and
                        function.__module__ not in ('core.reproducible',__name__) and
                        not function.__name__.startswith('_'))
            if core_alias or name in REPORT_COMPUTATIONS.get(module_name,()):
                setattr(module,name,base.recorded(function))


def begin(page,namespace=None,previous=None,others=()):
    recorder=base.begin(page,namespace,previous,others)
    instrument(namespace)
    return recorder


def replay_manifest(recorder):
    manifest=recorder.manifest()
    modules={c['module'] for c in manifest['calls']}
    if 'ui.charts' in modules:
        for relative in ('ui/charts.py',):
            manifest['runtime_sha256'][relative]=hashlib.sha256((base.ROOT/relative).read_bytes()).hexdigest()
    # A raw-core replay must carry the very same reference gate as the UI.
    if modules & {'core.dif_center','core.dif_lord_rasch'}:
        for relative in ('validation/lord_rasch_free_scale/results/report.json',
                         'tests/fixtures/irt_b10/data_dif1pl_uniform.csv'):
            path=base.ROOT/relative
            if path.is_file():manifest['runtime_sha256'][relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest


def python_code(manifest):
    code=base.python_code(manifest)
    if code is None:return None
    # Same existing pinned source fallback, with an exact manifest allowlist.
    old='if not parts or parts[0] not in ("core", "reports") or ".." in parts or not item.filename.endswith(".py"):'
    new='if not parts or ".." in parts or "/".join(parts) not in MANIFEST["runtime_sha256"]:'
    return code.replace(old,new)


def bundle(manifest,python):
    payload=io.BytesIO(base.bundle(manifest,python))
    with zipfile.ZipFile(payload,'a',zipfile.ZIP_DEFLATED) as archive:
        present=set(archive.namelist())
        for relative in manifest['runtime_sha256']:
            name='runtime/'+relative
            if name not in present:archive.write(base.ROOT/relative,name)
    return payload.getvalue()
