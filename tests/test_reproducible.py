"""Actual portable replay, isolation, stale-input and unsupported mapping checks."""
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

from core import descriptives
from core.reproducible import (Recorder, Unsupported, begin, bundle, decode, encode,
                               finish, python_code, recorded)
from core.reproducible_r import r_code


def sample():
    # Deterministic observed fixture; no simulation, fit or network.
    return pd.DataFrame({'M1': [1., 2., 3., np.nan, 5.], 'M2': [0., 1., 1., 0., 1.]}, index=[11, 12, 13, 14, 15])


def captured(frame):
    rec = begin('3B. Betimsel İstatistikler')
    try:
        result = descriptives.describe_items(frame)
        descriptives.describe_total(frame)
    finally:
        finish()
    return rec, result


def test_codec_preserves_real_values_types_labels_missing_and_index():
    x = sample()
    x['nullable'] = pd.Series([1, pd.NA, 2, 3, 4], index=x.index, dtype='Int64')
    x['label'] = ['A', 'B', None, 'ş', '1']
    payload = json.loads(json.dumps(encode(x), allow_nan=False))
    pd.testing.assert_frame_equal(decode(payload), x)
    a = np.array([[1., np.nan], [np.inf, -np.inf]])
    np.testing.assert_array_equal(decode(encode(a)), a)


def test_standalone_zip_executes_actual_prepared_data_outside_repository(tmp_path):
    rec, expected = captured(sample())
    manifest = rec.manifest()
    code = python_code(manifest)
    compile(code, 'analysis.py', 'exec')
    assert manifest['status'] == 'GENERATED / UNVERIFIED'
    r, reason = r_code(manifest)
    assert r and reason is None
    with zipfile.ZipFile(io.BytesIO(bundle(manifest, code, r))) as archive:
        archive.extractall(tmp_path)
    # Add comparison only in the test; no production execution/VERIFIED claim.
    script = tmp_path / 'analysis.py'
    script.write_text(code + '\nresults[0].to_json("actual.json", orient="split", double_precision=15)\n')
    env = dict(os.environ, PYTHONPATH='')
    run = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    actual = pd.read_json(tmp_path / 'actual.json', orient='split')
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    assert (tmp_path / 'analysis.R').is_file()
    assert 'NOT INSTALLED' not in (tmp_path / 'requirements-repro.txt').read_text()


def test_same_shape_changes_produce_distinct_code_and_no_previous_session_data():
    original = sample()
    first, _ = captured(original)
    changed = original.copy()
    changed.iloc[0, 0] = 4
    second, _ = captured(changed)
    assert python_code(first.manifest()) != python_code(second.manifest())
    third = begin('another analysis')
    finish()
    assert not third.calls
    assert third.manifest()['status'] == 'UNSUPPORTED'


def test_nested_calls_not_duplicated_and_exceptions_preserved():
    @recorded
    def inner(x):
        return x.mean()
    @recorded
    def outer(x, option='actual'):
        return inner(x)
    rec = begin('nested')
    try:
        pd.testing.assert_series_equal(outer(sample()), sample().mean())
    finally:
        finish()
    assert len(rec.calls) == 1
    args = decode(rec.calls[0]['arguments'])
    assert args['option'] == 'actual'
    @recorded
    def broken(x):
        raise ValueError('original estimator failure')
    begin('failure')
    try:
        with pytest.raises(ValueError, match='original estimator failure'):
            broken(sample())
    finally:
        assert not finish().calls


def test_opaque_configuration_is_unsupported_not_fabricated():
    @recorded
    def analysis(x, engine):
        return 7
    rec = begin('opaque')
    try:
        assert analysis(sample(), object()) == 7
    finally:
        finish()
    assert rec.errors and rec.manifest()['status'] == 'UNSUPPORTED'
    assert python_code(rec.manifest()) is None
    with pytest.raises(Unsupported):
        encode(pd.Categorical(['A', 'B']))


def test_unmatched_r_engine_never_substituted():
    manifest = {'calls': [{'module': 'core.irt_multidim', 'function': 'fit', 'arguments': encode({'data': sample(), 'model': 'GRM'})}]}
    code, reason = r_code(manifest)
    assert code is None and 'not established' in reason


def test_r_real_execution_and_basic_result_comparison(tmp_path):
    if shutil.which('Rscript') is None:
        if os.environ.get('REQUIRE_REPRO_R') == '1':
            pytest.fail('Required actual R runtime is missing')
        pytest.skip('NOT EXECUTED: actual R runtime unavailable')
    rec, expected = captured(sample())
    code, reason = r_code(rec.manifest())
    assert reason is None
    script = tmp_path / 'analysis.R'
    script.write_text(code)
    run = subprocess.run(['Rscript', str(script)], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    (tmp_path / 'R.stdout.log').write_text(run.stdout)
    (tmp_path / 'R.stderr.log').write_text(run.stderr)
    assert run.returncode == 0, run.stderr
    actual = pd.read_csv(tmp_path / 'describe_items.csv')
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-12, atol=1e-12)


def test_streamlit_code_panel_actual_inputs_and_supported_downloads():
    from streamlit.testing.v1 import AppTest
    script = '''
import pandas as pd
import streamlit as st
from core import descriptives
from core.reproducible import begin, finish
from ui.reproducible_view import render
value = st.number_input('First value', value=1.0)
begin('descriptive')
x = pd.DataFrame({'M1':[value,2.,3.,4.], 'M2':[0.,1.,0.,1.]})
descriptives.describe_items(x)
render(finish())
'''
    at = AppTest.from_string(script).run(timeout=30)
    assert not at.exception
    assert any(t.label == '</> Python / R Kodları' for t in at.tabs)
    before = at.code[0].value
    assert 'GENERATED / UNVERIFIED' in before
    at.number_input[0].set_value(8).run(timeout=30)
    assert not at.exception and at.code[0].value != before
    buttons = at.get('download_button')
    assert len(buttons) == 3
    assert all('analysis' in b.proto.url or b.proto.url for b in buttons)


def test_varargs_replay_preserves_signature_binding():
    import inspect
    def f(x, /, *extra, flag=1, **options):
        return x, extra, flag, options
    args = {'x': 2, 'extra': (3, 4), 'flag': 7, 'options': {'real': True}}
    bound = inspect.signature(f).bind_partial()
    bound.arguments.update(decode(encode(args)))
    assert f(*bound.args, **bound.kwargs) == (2, (3, 4), 7, {'real': True})


def test_cached_opaque_result_imports_only_actual_producer_no_refit():
    class Fit:
        pass
    calls = []
    @recorded
    def fit(x, setting):
        calls.append(setting)
        return Fit()
    @recorded
    def report(outcome, label):
        return pd.DataFrame({'label': [label]})
    first = begin('model')
    outcome = fit(sample(), setting='original')
    finish()
    second = begin('model', previous=first)
    report(outcome, 'actual cached result')
    finish()
    assert calls == ['original']  # No new calibration for code generation.
    assert [c['function'] for c in second.calls] == ['fit', 'report']
    assert second.calls[1]['arguments']['values'][0][1] == {'@': 'ref', 'path': [0]}
    assert second.previous is None
    other = begin('different page', previous=first)
    finish()
    assert not other.calls


def test_inventory_is_current_and_includes_missing_data_page():
    from tools.inventory_reproducible import inventory
    expected = inventory()
    path = Path(__file__).resolve().parents[1] / 'validation' / 'REPRODUCIBLE_ANALYSIS_INVENTORY.json'
    assert json.loads(path.read_text()) == expected
    assert any(r['screen'].startswith('pages/') for r in expected)


def test_r_unmapped_column_labels_do_not_break_original_analysis():
    frame = sample()
    frame.columns = [1, 2]
    rec, actual = captured(frame)
    assert len(actual) == 2
    r, reason = r_code(rec.manifest())
    assert r is None and 'UNSUPPORTED' in reason


def test_real_app_descriptives_current_settings_and_input_change():
    from streamlit.testing.v1 import AppTest
    root = Path(__file__).resolve().parents[1]
    at = AppTest.from_file(str(root / 'app.py'), default_timeout=30)
    frame = sample()
    for key, value in {'raw_df': frame, 'scored_df': frame.copy(), 'item_cols': list(frame),
                       'nav_page': '3B. Betimsel İstatistikler', 'id_col': None, 'group_col': None}.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert any(t.label == '</> Python / R Kodları' for t in at.tabs)
    code = next(c.value for c in at.code if 'MANIFEST = ' in c.value)
    assert 'describe_items' in code and 'describe_total' in code
    changed = frame.copy()
    changed.iloc[0, 0] = 9.
    at.session_state['scored_df'] = changed
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert next(c.value for c in at.code if 'MANIFEST = ' in c.value) != code
    at.session_state['scored_df'] = None
    at.session_state['raw_df'] = None
    at.run()
    assert not at.exception
    assert not at.code  # No stale downloadable script after data removal.


def test_preflight_only_never_exports_an_analysis_or_stale_download():
    from core import dimensionality
    rec = begin('Boyutluluk ve Faktör Analizi')
    dimensionality.data_type(sample())
    finish()
    assert rec.calls and rec.manifest()['status'] == 'UNSUPPORTED'
    assert python_code(rec.manifest()) is None
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string('''
import pandas as pd
from core import dimensionality
from core.reproducible import begin, finish
from ui.reproducible_view import render
begin('preflight only')
dimensionality.data_type(pd.DataFrame({'M1':[0,1], 'M2':[1,0]}))
render(finish())
''').run()
    assert not at.exception and not at.get('download_button')
