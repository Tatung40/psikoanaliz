"""Opt-in capture of actual public core calls; never changes calculation results.

No tracing, estimator reimplementation, automatic execution or dependency installs.
The recorder is session-local; wrappers remain inert outside a capture context.
"""
from contextvars import ContextVar
from functools import wraps
import hashlib
import importlib
import importlib.metadata
import inspect
import io
import json
import math
import re
import subprocess
from functools import lru_cache
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd

_ACTIVE = ContextVar('reproducible_capture', default=None)
_BUSY = ContextVar('reproducible_busy', default=False)
ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ONLY = frozenset({
    'fingerprint', 'data_type', 'prepare', 'prepare_polytomous', 'prepare_data',
    'validate', 'validate_inputs', 'validate_analysis_inputs', 'screen_items',
    'structure', 'specification', 'support_matrix', 'group_candidates', 'category_mapping',
})


class Unsupported(ValueError):
    """A value cannot be faithfully represented by the portable codec."""


def encode(value, references=None):
    """Lossless JSON representation; no pickle, repr/eval or lossy float CSV."""
    references = references or {}
    if id(value) in references:
        reference = references[id(value)]
        if isinstance(reference, dict):
            snapshot = encode(value)
            # In-place data edits invalidate the producer reference, even when
            # the object's identity, columns and shape are unchanged.
            if snapshot != reference['snapshot']:
                return snapshot
            return {'@': 'ref', 'path': reference['path'], 'snapshot': snapshot}
        return {'@': 'ref', 'path': reference}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if value is pd.NA:
        return {'@': 'NA'}
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else {'@': 'float', 'value': str(float(value))}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return {'@': 'timestamp', 'value': value.isoformat()}
    if isinstance(value, pd.MultiIndex):
        raise Unsupported('MultiIndex requires a dedicated mapping')
    if isinstance(value, pd.Index):
        return {'@': 'index', 'values': encode(value.tolist()), 'name': encode(value.name), 'dtype': str(value.dtype)}
    if isinstance(value, pd.DataFrame):
        if value.size > 250_000:
            raise Unsupported('Portable input exceeds 250000 cells; analysis remains unchanged')
        if any(isinstance(t, pd.CategoricalDtype) for t in value.dtypes):
            raise Unsupported('Categorical dtype requires explicit category/order metadata')
        return {'@': 'frame', 'columns': encode(value.columns), 'index': encode(value.index),
                'dtypes': [str(t) for t in value.dtypes], 'values': encode(value.to_numpy(dtype=object).tolist())}
    if isinstance(value, pd.Series):
        if isinstance(value.dtype, pd.CategoricalDtype):
            raise Unsupported('Categorical dtype requires explicit category/order metadata')
        return {'@': 'series', 'name': encode(value.name), 'index': encode(value.index),
                'dtype': str(value.dtype), 'values': encode(value.tolist())}
    if isinstance(value, np.ndarray):
        if value.size > 250_000:
            raise Unsupported('Portable array exceeds 250000 cells; analysis remains unchanged')
        if value.dtype.kind not in 'biufUSO':
            raise Unsupported(f'Array dtype {value.dtype}')
        return {'@': 'array', 'dtype': str(value.dtype), 'shape': list(value.shape), 'values': encode(value.tolist())}
    if isinstance(value, (list, tuple)):
        return {'@': 'tuple' if isinstance(value, tuple) else 'list', 'values': [encode(v, references) for v in value]}
    if isinstance(value, dict):
        return {'@': 'dict', 'values': [[encode(k), encode(v, references)] for k, v in value.items()]}
    raise Unsupported(f'{type(value).__module__}.{type(value).__name__}: no faithful public serialization')


def decode(value, results=(), *, reference_snapshots=False):
    if not isinstance(value, dict) or '@' not in value:
        return value
    tag = value['@']
    if tag == 'ref':
        if reference_snapshots:
            if 'snapshot' not in value:
                raise Unsupported('Opaque producer has no native-R data snapshot')
            return decode(value['snapshot'], reference_snapshots=True)
        result = results[value['path'][0]]
        for key in value['path'][1:]:
            result = result[key]
        return result
    if tag == 'NA':
        return pd.NA
    if tag == 'float':
        return float(value['value'])
    if tag == 'timestamp':
        return pd.Timestamp(value['value'])
    if tag in ('list', 'tuple'):
        items = [decode(v, results, reference_snapshots=reference_snapshots) for v in value['values']]
        return tuple(items) if tag == 'tuple' else items
    if tag == 'dict':
        return {decode(k, results, reference_snapshots=reference_snapshots): decode(v, results, reference_snapshots=reference_snapshots) for k, v in value['values']}
    if tag == 'index':
        return pd.Index(decode(value['values']), dtype=value['dtype'], name=decode(value['name']))
    if tag == 'frame':
        out = pd.DataFrame(decode(value['values']), index=decode(value['index']), columns=decode(value['columns']))
        for i, dtype in enumerate(value['dtypes']):
            out.isetitem(i, out.iloc[:, i].astype(dtype))
        return out
    if tag == 'series':
        return pd.Series(decode(value['values']), index=decode(value['index']), name=decode(value['name']), dtype=value['dtype'])
    if tag == 'array':
        return np.asarray(decode(value['values']), dtype=value['dtype']).reshape(value['shape'])
    raise Unsupported(f'Unknown wire tag: {tag}')


def _has_data(value):
    if isinstance(value, (pd.DataFrame, pd.Series, np.ndarray)):
        return True
    if isinstance(value, dict):
        return any(_has_data(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_data(v) for v in value)
    return False


def _remember(value, path, refs, keep):
    if isinstance(value, (pd.DataFrame, pd.Series, np.ndarray)):
        try:
            snapshot = encode(value)
        except (Unsupported, TypeError, ValueError, RecursionError, OverflowError):
            return
        refs[id(value)] = {'path': path, 'snapshot': snapshot}
        keep.append(value)
        return
    if id(value) in refs:
        return
    # Keep object references alive: Python may otherwise reuse their identities.
    if isinstance(value, (dict, list, tuple, pd.DataFrame, pd.Series, np.ndarray)) or not isinstance(value, (str, int, float, bool, type(None), np.generic)):
        refs[id(value)] = path
        keep.append(value)
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, (str, int)):
                _remember(item, path + [key], refs, keep)
    elif isinstance(value, (tuple, list)):
        for i, item in enumerate(value):
            _remember(item, path + [i], refs, keep)


class Recorder:
    def __init__(self, page, previous=None):
        self.page, self.calls, self.errors = page, [], []
        self.previous = previous if previous is not None and previous.page == page else None
        self.imported = {}
        self.references, self.keep = {}, []

    def _import_dependencies(self, value):
        if self.previous is None or isinstance(value, (pd.DataFrame, pd.Series, np.ndarray)):
            return
        if id(value) in self.previous.references and id(value) not in self.references:
            old_path = self.previous.references[id(value)]
            new_index = self._import_call(old_path[0])
            _remember(value, [new_index] + old_path[1:], self.references, self.keep)
        elif isinstance(value, dict):
            for item in value.values():
                self._import_dependencies(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self._import_dependencies(item)

    def _import_call(self, old_index):
        if old_index in self.imported:
            return self.imported[old_index]
        def remap(value):
            if isinstance(value, dict):
                if value.get('@') == 'ref':
                    path = value['path']
                    return {'@': 'ref', 'path': [self._import_call(path[0])] + path[1:]}
                return {k: remap(v) for k, v in value.items()}
            if isinstance(value, list):
                return [remap(v) for v in value]
            return value
        call = remap(self.previous.calls[old_index])
        index = len(self.calls)
        self.calls.append(call)
        self.imported[old_index] = index
        return index

    def invoke(self, function, args, kwargs, identity=None):
        token = _BUSY.set(True)
        try:
            payload, error, relevant = None, None, True
            try:
                bound = inspect.signature(function).bind(*args, **kwargs)
                bound.apply_defaults()
                self._import_dependencies(bound.arguments)
                relevant = _has_data(bound.arguments) or any(id(v) in self.references for v in bound.arguments.values())
                if relevant:
                    payload = {"arguments": encode(dict(bound.arguments), self.references)}
            except (Unsupported, TypeError, ValueError, RecursionError, OverflowError) as exc:
                error = str(exc)
            result = function(*args, **kwargs)  # Return/exception semantics are unchanged.
            if relevant:
                if error:
                    self.errors.append(f'{function.__module__}.{function.__name__}: {error}')
                else:
                    i = len(self.calls)
                    module, name = identity or (function.__module__, function.__name__)
                    self.calls.append({'module': module, 'function': name, **payload})
                    _remember(result, [i], self.references, self.keep)
            return result
        finally:
            _BUSY.reset(token)

    def manifest(self):
        versions = {}
        requirements = (ROOT / 'requirements.txt').read_text().splitlines()
        packages = [re.split(r'[<>=!~\[; ]', line.strip())[0] for line in requirements
                    if line.strip() and not line.lstrip().startswith('#')]
        for package in packages:
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = 'NOT INSTALLED'
        completed = any(call['function'] not in PREFLIGHT_ONLY for call in self.calls)
        errors = list(self.errors)
        if any(call['module'] == 'core.irt_dif34_bootstrap' for call in self.calls):
            errors.append('UNSUPPORTED: bootstrap DIF inference is gated by the UI; a raw-core replay must not expose suppressed p-values.')
        if not completed:
            errors.append('No completed calculation: preflight-only calls cannot export an analysis recipe.')
        return {'schema': 1, 'page': self.page, 'source_commit': source_commit(), 'runtime_sha256': runtime_hashes(),
                'python': sys.version.split()[0], 'packages': versions, 'calls': self.calls,
                'unsupported': errors,
                'preparation_history': 'UNAVAILABLE: exact post-preparation inputs are preserved; raw preprocessing steps are not guessed.',
                'input_stage': 'Exact inputs at public computation API boundary, after any UI filtering/recoding. Do not reapply preprocessing.',
                'seed': 'Only explicit seed/random_state arguments below; no invented seed or prior.',
                'status': 'GENERATED / UNVERIFIED' if completed and not errors else 'UNSUPPORTED'}


def recorded(function):
    if getattr(function, '__repro_recorded__', False):
        return function
    @wraps(function)
    def wrapper(*args, **kwargs):
        recorder = _ACTIVE.get()
        if recorder is None or _BUSY.get():
            return function(*args, **kwargs)
        return recorder.invoke(function, args, kwargs)
    wrapper.__repro_recorded__ = True
    return wrapper


def instrument(namespace=None):
    """Wrap existing public callables/aliases without changing any core source."""
    spaces = [namespace] if namespace is not None else []
    spaces += [vars(m) for name, m in list(sys.modules.items()) if m is not None and (name.startswith('core.') or name.startswith('ui.')) and name != __name__]
    for space in spaces:
        for name, function in list(space.items()):
            if inspect.isfunction(function) and function.__module__.startswith('core.') and function.__module__ != __name__ and not function.__name__.startswith('_'):
                space[name] = recorded(function)


def capture_view(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        if _ACTIVE.get() is not None:
            instrument()
        return function(*args, **kwargs)
    return wrapper


def begin(page, namespace=None, previous=None):
    recorder = Recorder(page, previous)
    _ACTIVE.set(recorder)
    instrument(namespace)
    return recorder


def public_call(module, name, *args, **kwargs):
    """Record an existing library operation used inline, without reimplementing it."""
    function = importlib.import_module(module)
    for part in name.split('.'):
        function = getattr(function, part)
    recorder = _ACTIVE.get()
    if recorder is None or _BUSY.get():
        return function(*args, **kwargs)
    return recorder.invoke(function, args, kwargs, identity=(module, name))


def finish():
    recorder = _ACTIVE.get()
    _ACTIVE.set(None)
    if recorder is not None:
        recorder.previous = None
    return recorder


@lru_cache(maxsize=1)
def source_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True, timeout=3).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return 'UNKNOWN; see exact runtime_sha256'


def runtime_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('core', 'reports') for p in sorted((ROOT / folder).glob('*.py'))
            if not p.name.startswith('reproducible')}


def python_code(manifest):
    """Standalone script with embedded exact inputs and optional JSON data path."""
    if manifest['status'] == 'UNSUPPORTED':
        return None
    # Embed the small codec, not an estimator. Runtime core is shipped in the ZIP.
    codec = '\n\n'.join(inspect.getsource(f) for f in (Unsupported, decode))
    data = json.dumps(manifest, ensure_ascii=False, allow_nan=False)
    return f'''# PsikoAnaliz reproducible analysis — GENERATED / UNVERIFIED
# Scientific equivalence/certification is not implied by code generation.
# Offline: extract ZIP; install requirements-repro.txt; run python analysis.py
# Without ZIP the pinned public calculation sources are fetched once (never your data).
# Input is the exact post-preparation analysis input. Original labels/settings are in JSON.
# Optional replacement: python analysis.py /path/to/analysis-input.json
import hashlib, importlib, inspect, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

{codec}

MANIFEST = json.loads({data!r})
if len(sys.argv) > 1:
    MANIFEST = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
# The adjacent runtime contains the pinned existing project calculation APIs.
RUNTIME = Path(__file__).resolve().parent / "runtime"
if not (RUNTIME / "core").is_dir():
    import io, tempfile, urllib.request, zipfile
    commit = MANIFEST["source_commit"]
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise SystemExit("No pinned source commit: extract the accompanying offline ZIP.")
    url = "https://codeload.github.com/Tatung40/psikoanaliz/zip/" + commit
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = response.read(50_000_001)
    if len(payload) > 50_000_000:
        raise SystemExit("Unexpected source archive size; use the offline ZIP.")
    RUNTIME = Path(tempfile.mkdtemp(prefix="psikoanaliz-runtime-"))
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for item in archive.infolist():
            parts = Path(item.filename).parts[1:]
            if not parts or parts[0] not in ("core", "reports") or ".." in parts or not item.filename.endswith(".py"):
                continue
            target = RUNTIME.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(item))
for relative, expected in MANIFEST["runtime_sha256"].items():
    if hashlib.sha256((RUNTIME / relative).read_bytes()).hexdigest() != expected:
        raise SystemExit("Calculation source hash mismatch: " + relative)
sys.path.insert(0, str(RUNTIME))
results = []
for call in MANIFEST["calls"]:
    function = importlib.import_module(call["module"])
    for part in call["function"].split("."):
        function = getattr(function, part)
    bound = inspect.signature(function).bind_partial()
    bound.arguments.update(decode(call["arguments"], results))
    result = function(*bound.args, **bound.kwargs)
    results.append(result)
    print("\\n", call["module"] + "." + call["function"])
    print(result)
print("Execution complete; numerical verification requires comparison to original outputs.")
'''


def bundle(manifest, python, r=None):
    """Offline calculation sources + exact inputs. Never uploads user data."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('analysis.py', python)
        archive.writestr('analysis-input.json', json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2))
        if r:
            archive.writestr('analysis.R', r)
        requirements = (ROOT / 'requirements.txt').read_text()
        pinned = '\n'.join(f'{k}=={v}' for k, v in manifest['packages'].items() if v != 'NOT INSTALLED')
        archive.writestr('requirements-repro.txt', requirements + '\n# Captured runtime versions\n' + pinned + '\n')
        # Existing local calculation modules, not an invented equivalent engine.
        for directory in ('core', 'reports'):
            for path in sorted((ROOT / directory).glob('*.py')):
                archive.write(path, 'runtime/' + str(path.relative_to(ROOT)))
        archive.writestr('README.txt', 'Exact prepared inputs are embedded in analysis.py and analysis-input.json.\n'
                         'Do not reapply category coding, filtering or missing-value preprocessing.\n'
                         'Install requirements-repro.txt in a virtual environment. No packages are installed automatically.\n'
                         'Python reuses the bundled PsikoAnaliz engine; this is not an independent reference.\n'
                         'Code generation is not scientific validation. R may be unsupported.\n')
    return output.getvalue()
