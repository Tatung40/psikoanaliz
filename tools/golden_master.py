"""Golden master of user-visible Python-library analysis outputs.

Equivalence gate for the removal of the native ("Eser") binary IRT engine: the
outputs of a REFERENCE checkout (main before the removal) and of the current tree
are captured on the same machine in the same CI job and compared exactly. A stored
master is not used: ill-conditioned EM fits (4PL boundary, multigroup GRM) stop at
environment-dependent points (~1e-5), so only same-environment comparison is valid.
Only library-backed screens are recorded; the retired native engine is not a
reference. Volatile fields (versions, platform, timestamps) are excluded.

    python tools/golden_master.py capture OUT.json.gz [--root CHECKOUT]
    python tools/golden_master.py compare REFERENCE.json.gz CURRENT.json.gz
"""
import gzip
import io
import json
import math
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VOLATILE = ('python', 'platform', 'version', 'versions', 'numpy', 'pandas', 'scipy', 'sürüm',
            'engine', 'motor', 'signature', 'timestamp', 'tarih', 'zaman', 'log')


def _volatile(key):
    k = str(key).casefold()
    return any(v in k for v in VOLATILE)


def encode(value):
    """Deterministic JSON-safe structure; floats kept at full precision."""
    if isinstance(value, pd.DataFrame):
        frame = value.reset_index() if not isinstance(value.index, pd.RangeIndex) else value
        cols = [str(c) for c in frame.columns]
        keep = [i for i, c in enumerate(cols) if not _volatile(c)]
        if cols[:1] in (['Setting'], ['Özellik']):  # key/value tables: drop volatile rows
            frame = frame[[not _volatile(k) for k in frame.iloc[:, 0]]]
        return {'__frame__': [cols[i] for i in keep],
                'rows': [[encode(row[i]) for i in keep] for row in frame.itertuples(index=False, name=None)]}
    if isinstance(value, pd.Series):
        return encode(value.to_frame())
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items() if not _volatile(k)}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    if isinstance(value, np.ndarray):
        return [encode(v) for v in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return {'__float__': 'nan' if math.isnan(v) else ('inf' if v > 0 else '-inf')} if not math.isfinite(v) else v
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    return str(value)


def _csv_frame(data):
    return pd.read_csv(io.BytesIO(data), encoding='utf-8-sig', keep_default_na=False, dtype=str)


def _xlsx_frames(data):
    book = pd.read_excel(io.BytesIO(data), sheet_name=None, dtype=str, keep_default_na=False)
    return {name: frame for name, frame in book.items()}


def downloads_semantic(files):
    """Semantic content of exported files: parsed tables, not container bytes."""
    out = {}
    for name, data in sorted(files.items()):
        ext = name.rsplit('.', 1)[-1].lower()
        if ext in ('csv',):
            out[name] = encode(_csv_frame(data))
        elif ext in ('tsv', 'txt'):
            out[name] = encode(pd.read_csv(io.BytesIO(data), sep='\t', encoding='utf-8-sig', dtype=str, keep_default_na=False))
        elif ext == 'xlsx':
            out[name] = encode(_xlsx_frames(data))
        elif ext == 'json':
            out[name] = encode(json.loads(data))
        elif ext == 'zip':
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                out[name] = downloads_semantic({n: archive.read(n) for n in archive.namelist()})
        elif ext == 'png':
            out[name] = {'png_bytes': len(data) > 100 and data[:8] == b'\x89PNG\r\n\x1a\n'}
        else:
            out[name] = data.decode('utf-8', errors='replace') if isinstance(data, bytes) else str(data)
    return out


def _example(model, family='irt', level=0):
    from core import example_datasets as ex
    spec = next(d for d in ex.registry().values() if d.family == family and d.model == model and d.level == level)
    return ex.materialize(spec.id)


def python_mirt_models():
    """Screen 'MTK: Python mirt <model>' (ui/irt_python_library_view.py) core outputs."""
    import mirt
    from core.irt_b14 import prepare_polytomous
    from core.irt_python_mirt_adapter import fit_python_mirt
    from core.irt_python_mirt_report import build_report, MODELS
    from core.irt_person_abilities import person_abilities
    from core.irt_rasch_wright import rasch_wright_tables
    from core.irt_library_diagnostics import library_fit_diagnostics
    out = {}
    for model in MODELS:
        bundle = _example('1PL' if model == 'Rasch' else model)
        x = bundle['data'][bundle['items']]
        labels = ()
        if model in ('GRM', 'PCM', 'GPCM', 'NRM'):
            x, labels = prepare_polytomous(x, model, categories=['A', 'B', 'C', 'D'] if model == 'NRM' else None)
        x = x.astype(int)
        cats = None if model in ('Rasch', '1PL', '2PL', '3PL', '4PL') else len(labels)
        outcome = fit_python_mirt(x.to_numpy(), model=model, n_categories=cats)
        report = build_report(outcome, model=model, items=list(x.columns), labels=labels, theta=np.linspace(-4, 4, 161))
        record = {'metadata': report.metadata, 'parameters': report.parameters, 'curves': report.curves,
                  'information': report.information, 'warnings': report.warnings, 'errors': report.errors,
                  'diagnostics': library_fit_diagnostics(outcome.fit)}
        ids = bundle['data'][bundle['id']].to_numpy()
        if model == 'Rasch':
            from core.irt_rasch_wright import wright_figure
            record['wright'] = rasch_wright_tables(outcome, x, item_names=list(x.columns), identifiers=ids)
            figure = wright_figure(record['wright']).to_plotly_json()
            record['wright_figure'] = {'data': figure['data'], 'layout': {k: figure['layout'].get(k) for k in ('title', 'xaxis', 'yaxis', 'barmode', 'height')}}
        else:
            record['persons'] = person_abilities(x.to_numpy(), report.curves, model=model, identifiers=ids)
        if model != 'NRM':
            record['itemfit'] = mirt.itemfit(outcome.fit, x.to_numpy(dtype=int), statistics=['infit', 'outfit', 'S_X2'])
        out[model] = record
    return out


def binary_main_page():
    """Screen 'MTK: 1PL..4PL' (app.py) library parameter table and curves."""
    from core.irt_python_mirt_adapter import fit_python_mirt
    from core.irt_python_mirt_parameters import python_mirt_binary_parameters
    from core.irt_library_parameter_mapping import curves_from_normalized_table
    from core.irt_binary_usage_policy import usage_status
    out = {}
    for model in ('1PL', '2PL', '3PL', '4PL'):
        bundle = _example(model)
        x = bundle['data'][bundle['items']].astype(int)
        fit = fit_python_mirt(x.to_numpy(), model=model).fit
        table = python_mirt_binary_parameters(fit, model=model, item_names=list(x.columns))
        curves = curves_from_normalized_table(table[['Madde', 'a', 'b', 'c', 'd']], np.linspace(-4, 4, 81))
        out[model] = {'items': table, 'curves': curves, 'usage': usage_status(model, 'ML', len(x), x.shape[1])}
    return out


def b14():
    from core.irt_b14 import fit_b14
    from core.irt_b14_persons_info import person_scores, information_table, model_comparison
    from core.irt_b14_profiles import empirical_profiles
    from reports.irt_b14_report import report_tables
    out = {}
    for model in ('GRM', 'PCM', 'GPCM', 'NRM'):
        bundle = _example(model)
        result = fit_b14(bundle['data'][bundle['items']], model,
                         categories=['A', 'B', 'C', 'D'] if model == 'NRM' else None)
        record = {'tables': report_tables(result), 'persons': person_scores(result),
                  'information': information_table(result),
                  'profiles': empirical_profiles(result['coded'], result['labels'], model)}
        if model == 'GRM':
            record['comparison'] = model_comparison(result['coded'])
        out[model] = record
    return out


def python_dif():
    from core import python_mirt_dif as P
    out = {}
    for model in ('2PL', 'GRM'):
        b = _example(model, family='dif', level=1)
        r = P.analyze(b['data'], items=b['items'], group=b['group'], reference='Referans', focal='Odak',
                      model=model, anchors=b['items'][:2], studied=b['items'][2:4], n_quadpts=21)
        out[model] = {'metadata': r['metadata'], 'tables': r['tables']}
    return out


def mirt_multidim():
    from core.irt_multidim import fit, report
    out = {}
    for model in ('2PL', 'GRM'):
        b = _example(model, family='mirt')
        items = b['items']
        dims = b['spec'].dimensions
        pattern = np.zeros((len(items), dims), int)
        pattern[np.arange(len(items)), np.minimum(np.arange(len(items)) * dims // len(items), dims - 1)] = 1
        result = report(fit(b['data'][items], model, pattern, n_quadpts=11), b['data'][b['id']].to_numpy())
        out[model] = {k: v for k, v in result.items() if isinstance(v, (pd.DataFrame, dict, list, str, int, float))}
    return out


def local_and_dimensionality():
    from core import local_fit
    from core.dimensionality import dimensionality_analysis
    b = _example('2PL')
    lf = local_fit.run(b['data'][b['items']].astype(int), model='2PL')
    f = _example('Sürekli iki faktör', family='factor')
    dim = dimensionality_analysis(f['data'][f['items']], n_factors=2, pa_reps=20)
    dim = {k: v for k, v in dim.items() if k != 'correlation'} | {'correlation': dim['correlation']['R']}
    return {'local_fit': {k: v for k, v in lf.items() if k != 'obs_exp'}, 'dimensionality': dim}


def generalizability():
    from core import example_datasets as ex
    from core import generalizability as g
    out = {}
    for d in ex.available(ex.G_PAGE):
        b = ex.materialize(d.id)
        facets = g.DESIGNS[d.design]
        df, counts = g.prepare(b['data'], d.design, {f: f for f in facets}, 'score')
        scenarios = {f: [n] if f == 'p' else [n, n + 2] for f, n in counts.items()}
        r = g.analyze(b['data'], d.design, {f: f for f in facets}, 'score', scenarios=scenarios)
        out[d.id] = g.report_tables(r, .8) | {'counts': r['counts']}
    return out


def ctt():
    from core import example_datasets as ex
    from core.ctt_dual_engine import compare_ctt, item_guidance
    out = {}
    for d in [x for x in ex.registry().values() if x.family == 'abc']:
        b = ex.materialize(d.id)
        s = ex.scored(b)
        r = compare_ctt(s, b['items'])
        out[d.id] = {'alpha': r['native_alpha'], 'reference_alpha': r['reference_alpha'], 'ci': r['ci'],
                     'item_stats': r['item_stats'], 'guidance': item_guidance(r), 'N': r['N']}
    return out


def example_library():
    from core import example_datasets as ex
    return {d: downloads_semantic(ex.downloads(ex.materialize(d))) for d in sorted(ex.registry())}


SECTIONS = {'python_mirt_models': python_mirt_models, 'binary_main_page': binary_main_page, 'b14': b14,
            'python_dif': python_dif, 'mirt_multidim': mirt_multidim,
            'local_and_dimensionality': local_and_dimensionality, 'generalizability': generalizability,
            'ctt': ctt, 'example_library': example_library}


def run_section(name):
    # Multigroup EM (tol 1e-4) is path dependent on BLAS reduction order at ~1e-5;
    # one BLAS thread makes the replay bit-identical on a given platform.
    from threadpoolctl import threadpool_limits
    warnings.filterwarnings('ignore')
    with threadpool_limits(limits=1):
        return encode(SECTIONS[name]())


def capture(names=None):
    return {name: run_section(name) for name in (names or SECTIONS)}


def save(data, path):
    with gzip.open(path, 'wt', encoding='utf-8', compresslevel=9) as handle:
        json.dump(data, handle, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def load(path):
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return json.load(handle)


def compare(expected, actual, path='', rtol=1e-6, atol=1e-8, out=None):
    """List every difference; floats use a fixed predeclared tolerance."""
    out = [] if out is None else out
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            out.append(f'{path}: keys {sorted(set(expected) ^ set(actual))[:6]}')
        for k in sorted(set(expected) & set(actual)):
            compare(expected[k], actual[k], f'{path}/{k}', rtol, atol, out)
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            out.append(f'{path}: length {len(expected)} != {len(actual)}')
        for i, (e, a) in enumerate(zip(expected, actual)):
            compare(e, a, f'{path}[{i}]', rtol, atol, out)
    elif isinstance(expected, float) and isinstance(actual, (float, int)) and not isinstance(actual, bool):
        if not math.isclose(expected, actual, rel_tol=rtol, abs_tol=atol):
            out.append(f'{path}: {expected!r} != {actual!r}')
    elif expected != actual:
        out.append(f'{path}: {str(expected)[:80]!r} != {str(actual)[:80]!r}')
    return out


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    cap = sub.add_parser('capture'); cap.add_argument('out'); cap.add_argument('--root')
    cmp_ = sub.add_parser('compare'); cmp_.add_argument('reference'); cmp_.add_argument('current')
    cmp_.add_argument('--rtol', type=float, default=0.); cmp_.add_argument('--atol', type=float, default=0.)
    args = parser.parse_args(argv)
    if args.command == 'capture':
        import sys
        if args.root:  # import core/ui/reports from another checkout (the reference commit)
            sys.path.insert(0, str(Path(args.root).resolve()))
            for name in [m for m in sys.modules if m.split('.')[0] in ('core', 'ui', 'reports')]:
                del sys.modules[name]
        data = capture()
        save(data, args.out)
        print(f'captured {list(data)} -> {args.out}')
        return 0
    reference, current = load(args.reference), load(args.current)
    differences = compare(reference, current, rtol=args.rtol, atol=args.atol)
    for line in differences[:50]:
        print(line)
    print(f'{len(differences)} differences in {len(reference)} sections (rtol={args.rtol}, atol={args.atol})')
    return 1 if differences or set(reference) != set(current) else 0


if __name__ == '__main__':
    raise SystemExit(main())
