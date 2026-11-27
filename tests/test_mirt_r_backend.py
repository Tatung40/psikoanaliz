import numpy as np, pandas as pd, pytest
from core import mirt_r_backend as RB

def binary(n=200, j=8, seed=0):
    r = np.random.default_rng(seed)
    return pd.DataFrame((r.random((n, j)) < .5).astype(int), columns=[f'I{i}' for i in range(j)])

PAT = np.array([[1, 0]] * 4 + [[0, 1]] * 4)

def test_whitelist_and_validation():
    x = binary()
    with pytest.raises(ValueError): RB.validate(x, 'system("rm -rf /")')
    with pytest.raises(ValueError): RB.validate(x, 'M3PL_fixed', pattern=PAT, c=.7)
    with pytest.raises(ValueError): RB.validate(x, 'M4PL_fixed', pattern=PAT, c=.2, u=.4)
    with pytest.raises(ValueError): RB.validate(x, 'M3PL_fixed', pattern=PAT, u=.9)
    with pytest.raises(ValueError): RB.validate(x, 'M3PL_fixed', pattern=np.ones((8, 2), int))  # no pure anchors
    with pytest.raises(ValueError): RB.validate(x, 'bfactor', groups=np.repeat([0, 1], 4), itemtype='Rasch')
    with pytest.raises(ValueError): RB.validate(x, 'M3PL_fixed', pattern=PAT, max_iter=10**6)
    r = RB.validate(x, 'M4PL_fixed', pattern=PAT, c=.2, u=.95)['request']
    assert r['c'] == [.2] * 8 and r['u'] == [.95] * 8 and r['spec'] == 'M4PL_fixed'

def test_item_names_are_data_not_code():
    x = binary(); x.columns = [f'x"); system("echo pwned"); ("{i}' for i in range(8)]
    req = RB.validate(x, 'M3PL_fixed', pattern=PAT, c=.2)['request']
    assert req['items'][0].startswith('x")')  # serialised verbatim to JSON/CSV, never evaluated

def test_explicit_refusal_without_engine(monkeypatch):
    monkeypatch.setattr(RB, 'engine_available', lambda: (False, 'R mirt is not installed'))
    with pytest.raises(RB.RBackendUnavailable, match='not installed'):
        RB.fit(binary(), 'M3PL_fixed', pattern=PAT, c=.2)

def test_script_has_no_install_or_eval():
    s = RB.SCRIPT.read_text()
    assert 'install' not in s and 'eval(' not in s and 'parse(' not in s and 'system(' not in s

@pytest.mark.skipif(not RB.engine_available()[0], reason='R mirt 1.48 not installed')
def test_actual_backend_fit_metadata():
    r = RB.fit(binary(300), 'M3PL_fixed', pattern=PAT, c=.15, n_quadpts=9, max_iter=200, tol=1e-6)
    assert r['engine'] == 'R mirt' and r['mirt_version'] == '1.48' and r['status'] == RB.STATUS
    v = r['values']; assert np.allclose(v[v.name == 'g'].value, .15) and not v[v.name == 'g'].est.any()
    assert r['EAP'].shape == (300, 2) and np.isfinite(r['common_grid_logLik'])
