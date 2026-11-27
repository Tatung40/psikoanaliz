import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd, pytest

ROOT = Path(__file__).resolve().parents[1]
MATRIX = json.loads((ROOT / 'validation/mirt_advanced/scope_matrix.json').read_text())['scopes']

def test_polytomous_bifactor_withheld_and_api_unsupported():
    mirt = pytest.importorskip('mirt')
    x = np.random.default_rng(0).integers(0, 3, (200, 9))
    with pytest.raises(Exception, match='dichotomous'):
        mirt.bfactor(x, [0, 0, 0, 1, 1, 1, 2, 2, 2], n_quadpts=5, max_iter=5, compute_standard_errors=False)
    for k in ('GRM_bifactor', 'GPCM_bifactor', 'PCM_bifactor'):
        assert MATRIX[k]['level'] == 'WITHHELD' and MATRIX[k]['decision'] == 'NO'

def test_production_rejects_asymptote_families_and_unvalidated_exploratory():
    M = pytest.importorskip('core.irt_multidim')
    x = pd.DataFrame((np.random.default_rng(1).random((100, 6)) < .5).astype(int), columns=list('abcdef'))
    pat = [[1, 0]] * 3 + [[0, 1]] * 3
    for m in ('3PL', '4PL'):
        with pytest.raises(ValueError):
            M.fit(x, m, pat, max_iter=2)
    for m in ('PCM', 'Rasch', 'NRM'):
        with pytest.raises(ValueError):
            M.fit(x, m, pat, exploratory=True, max_iter=2)
    assert M.STATUS == 'EXPERIMENTAL / NOT CERTIFIED'

def test_no_scope_claims_certification_or_unavailable_r_backend():
    assert all('CERTIFIED' not in v.get('level', '') for v in MATRIX.values())
    assert MATRIX['R_backend']['available_to_users'] == 'NO'
    for k in ('M3PL_free_asymptote', 'M4PL_free_asymptotes', 'M4PL_mixed_asymptotes'):
        assert MATRIX[k]['level'] == 'WITHHELD'

def test_stage1_criteria_unchanged():
    text = (ROOT / 'validation/mirt_stage1/criteria.json').read_bytes()
    assert hashlib.sha256(text).hexdigest() == CRITERIA_SHA

CRITERIA_SHA = '5b666c899ee9cc53e3683b99fe6d192816fdcc0731ca0a32b61607f612910d44'
