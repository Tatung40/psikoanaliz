"""Actual UI/export integration for the advanced MIRT routes.

These are software fixtures, not a new calibration campaign. Scientific evidence lives in validation/mirt_advanced.
"""
import io, os, zipfile
from pathlib import Path
import numpy as np, pandas as pd, pytest
from streamlit.testing.v1 import AppTest
from core import mirt_advanced_report as advr, mirt_r_backend as rb
from reports.irt_multidim_report import downloads

OUT = Path(os.environ.get('MIRT_ADV_ARTIFACTS', '/tmp/mirt_adv_artifacts'))


def poly(model='GPCM', n=600, groups=(0, 0, 0, 1, 1, 1, 2, 2, 2), seed=7):
    import mirt
    from mirt.models.polytomous import GeneralizedPartialCredit, GradedResponseModel
    g = np.array(groups); P = advr.adv.bifactor_pattern(g); k, q = P.shape
    e = (GeneralizedPartialCredit if model == 'GPCM' else GradedResponseModel)(k, n_factors=q, n_categories=3)
    e.set_parameters(discrimination=P * 1.2, **{'steps' if model == 'GPCM' else 'thresholds': np.tile([-.5, .5], (k, 1))})
    x = mirt.simdata(model=e, n_persons=n, theta=np.random.default_rng(seed).standard_normal((n, q)), seed=seed)
    return pd.DataFrame(x, columns=[f'M{i+1}' for i in range(k)]), g


def check_bundle(result, name):
    meta, tables = result['metadata'], result['tables']
    assert meta['Status'] == 'EXPERIMENTAL / NOT CERTIFIED' and meta['Inferential_status'] == 'WITHHELD'
    scope = tables['Output_scope']
    assert set(scope.query('Inferential_status == "WITHHELD"').Output) >= set(advr.WITHHELD)
    assert not any(c in t.columns for t in tables.values() for c in ('SE', 'CI_lower', 'CI_upper', 'p_value'))
    item = tables['Parameters'].iloc[0, 0]; cat = meta['Labels'][-1]
    bundle = downloads(result, item, cat)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('csv', 'xlsx', 'png', 'zip'):
        if bundle[ext] is not None:
            (OUT / f'{name}.{ext}').write_bytes(bundle[ext] if isinstance(bundle[ext], bytes) else bundle[ext].encode())
    assert bundle['csv'] and bundle['xlsx'] and bundle['zip']
    with zipfile.ZipFile(io.BytesIO(bundle['zip'])) as z:
        names = z.namelist(); text = ' '.join(z.read(n).decode('utf-8', 'ignore') for n in names if n.endswith(('.csv', '.md', '.json', '.txt')))
    assert 'EXPERIMENTAL / NOT CERTIFIED' in text and 'WITHHELD' in text
    pd.read_excel(io.BytesIO(bundle['xlsx']), sheet_name=None)
    return bundle


def test_gpcm_bifactor_report_and_exports():
    x, g = poly('GPCM')
    r = advr.bifactor_report(x, g, list(advr.BIFACTOR_MODELS)[1], n_quadpts=7, max_iter=1000, tol=1e-7)
    assert r['metadata']['Converged'] and r['metadata']['High_precision'] is False
    assert {'EAP', 'Parameters', 'Covariance'} <= set(r['tables'])
    check_bundle(r, 'gpcm_bifactor')


def test_grm_bifactor_high_precision_report_and_exports():
    x, g = poly('GRM')
    r = advr.bifactor_report(x, g, list(advr.BIFACTOR_MODELS)[0], high_precision=True, n_quadpts=7)
    assert r['metadata']['High_precision'] is True and r['metadata']['Continuation_seconds'] > 0
    check_bundle(r, 'grm_bifactor_hp')


def test_grm_high_precision_profile_report():
    import mirt
    from mirt.models.polytomous import GradedResponseModel
    pat = np.ones((8, 2), int); pat[0, 1] = 0
    e = GradedResponseModel(8, n_factors=2, n_categories=4); e.set_parameters(discrimination=pat * 1.1, thresholds=np.tile([-1, 0, 1], (8, 1)))
    x = pd.DataFrame(mirt.simdata(model=e, n_persons=300, theta=np.random.default_rng(1).standard_normal((300, 2)), seed=2), columns=[f'J{i}' for i in range(8)])
    r = advr.grm_high_precision_report(x, pat, exploratory=True, n_quadpts=11)
    d = r['tables']['Precision_diagnostics'].iloc[0]
    assert d.continuation_seconds > 0 and 'NOT validated' in d.note
    check_bundle(r, 'grm_efa_high_precision')


def test_pcm_bifactor_and_free_asymptote_r_refused():
    x, g = poly('GPCM')
    with pytest.raises(ValueError, match='WITHHELD'):
        advr.adv.fit_polytomous_bifactor(x, g, 'PCM')
    with pytest.raises(ValueError, match='fixed-asymptote'):
        advr.native_r_report(x, '3PL — 2D between-item, c serbest (SciPy)', np.eye(9, 2, dtype=int))
    with pytest.raises(ValueError, match='exactly 3'):
        advr.bifactor_report(x.iloc[:, :8], np.repeat([0, 1], 4), list(advr.BIFACTOR_MODELS)[1])


def test_native_r_refuses_clearly_when_unavailable(monkeypatch):
    monkeypatch.setattr(rb, 'engine_available', lambda: (False, 'R mirt is not installed; the native R backend is unavailable (no runtime installation).'))
    d = pd.read_csv('validation/mirt_phase3/fixtures/2PL_2D.csv'); P = np.zeros((d.shape[1], 2), int); P[:4, 0] = P[8:, 0] = 1; P[4:8, 1] = 1
    with pytest.raises(rb.RBackendUnavailable, match='not installed'):
        advr.native_r_report(d, '4PL — genel pozitif CFA, c/d sabit (SciPy)', P, c=.2, u=.95)


@pytest.mark.skipif(not rb.engine_available()[0], reason='R mirt 1.48 not installed')
def test_native_r_fixed_4pl_report_and_exports():
    d = pd.read_csv('validation/mirt_phase3/fixtures/2PL_2D.csv'); P = np.zeros((d.shape[1], 2), int); P[:4, 0] = P[8:, 0] = 1; P[4:8, 1] = 1
    r = advr.native_r_report(d, '4PL — genel pozitif CFA, c/d sabit (SciPy)', P, c=.2, u=.95, n_quadpts=15, max_iter=3000, tol=1e-8)
    m = r['metadata']; assert m['Backend'] == 'NATIVE_R_MIRT' and m['Converged']
    assert abs(m['LogLik'] - (-2536.18391098393)) < 1e-6 and abs(m['Common_grid_logLik'] - m['LogLik']) < 1e-6
    check_bundle(r, 'native_r_m4pl_fixed')


def test_actual_ui_gpcm_bifactor_flow():
    x, g = poly('GPCM'); x.insert(0, 'ID', [f'P{i}' for i in range(len(x))]); items = list(x.columns[1:])
    at = AppTest.from_file(str(Path('app.py').resolve()), default_timeout=300)
    for k, v in dict(raw_df=x, scored_df=None, item_cols=items, id_col='ID', nav_page='Çok Boyutlu MTK (MIRT)').items():
        at.session_state[k] = v
    at.run(); assert not at.exception
    at.selectbox(key='md_model').set_value(list(advr.BIFACTOR_MODELS)[1]).run(); assert not at.exception
    at.text_input(key='md_groups_4').set_value(','.join(map(str, g))).run(); assert not at.exception
    at.button(key='md_run').click().run(); assert not at.exception
    assert any('EXPERIMENTAL / NOT CERTIFIED' in i.value for i in at.info)
    assert at.get('download_button')


def test_grm_bifactor_default_profile_never_claims_agreement_and_actual_docx(monkeypatch):
    """Actual academic DOCX through the UI path; GRM bifactor with high precision disabled."""
    import io
    from docx import Document
    x, g = poly('GRM'); x.insert(0, 'ID', [f'P{i}' for i in range(len(x))]); items = list(x.columns[1:])
    page = 'Çok Boyutlu MTK (MIRT)'
    at = AppTest.from_file(str(Path('app.py').resolve()), default_timeout=600)
    for k, v in dict(raw_df=x, scored_df=None, item_cols=items, id_col='ID', nav_page=page).items():
        at.session_state[k] = v
    at.run(); at.selectbox(key='md_model').set_value(list(advr.BIFACTOR_MODELS)[0]).run()
    at.text_input(key='md_groups_4').set_value(','.join(map(str, g))).run()
    at.checkbox(key='md_bf_hp').set_value(False).run()
    at.button(key='md_run').click().run(); assert not at.exception
    raw = at.session_state['_academic_history'][page]['docx']; doc = Document(io.BytesIO(raw))
    text = '\n'.join([p.text for p in doc.paragraphs] + [c.text for t in doc.tables for r in t.rows for c in r.cells])
    OUT.mkdir(parents=True, exist_ok=True); (OUT / 'grm_bifactor_default_academic.docx').write_bytes(raw)
    for needed in ('EXPERIMENTAL / NOT CERTIFIED', 'WITHHELD', 'Item SE/CI', 'core.mirt_advanced.fit_polytomous_bifactor', 'PCM bifactor WITHHELD', 'NOT VALIDATED'):
        assert needed in text, needed
    assert 'single-fixture R agreement' not in text


def test_native_r_metadata_reaches_report_tables(monkeypatch):
    src = Path('core/mirt_advanced_report.py').read_text()
    assert 'Backend="NATIVE_R_MIRT"' in src and 'Validation_scope=' in src and '"Summary"' in src
