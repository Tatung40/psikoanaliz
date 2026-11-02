"""Read-only independent R evidence checks; no fits or dependencies installed."""
import hashlib,json
from pathlib import Path


def test_real_r_reference_is_scoped_matches_sources_and_meets_predeclared_chi_gate():
    evidence=json.loads(Path('validation/DIF_WALD_LORD_REFERENCE.json').read_text())
    assert evidence['R_execution']=='EXECUTED' and evidence['numerical_status']=='PASS'
    assert evidence['chi_relative_limit']==1e-3
    assert evidence['status']=='MEASURED / NOT CERTIFIED'
    for source,sha in evidence['source_sha256'].items():
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==sha
    assert set(evidence['cases'])=={'dif1pl_uniform','dif2pl_mixed','dif2pl_none'}
    for case,record in evidence['cases'].items():
        settings=json.loads(Path('validation/dif_wald_lord/results',case,'settings.json').read_text())
        assert record['R']['data_sha256']==settings['data_sha256']
        assert record['R']['used_n']==settings['used_n']
        assert record['R']['mirt']=='1.48'
        assert record['R']['converged']==[True,True,True]
        assert record['R']['bounds_interior']==[True,True,True]
        for method in ('wald','lord'):
            result=record['comparisons'][method]
            assert result['status']=='PASS'
            assert result['chi2']['max_relative']<=1e-3
            assert result['chi2']['undefined_relative_count']==0
            assert result['covariance']['max_absolute']>=0


def test_actual_r_cache_rejects_changed_source_or_fit_data(monkeypatch):
    from validation.dif_wald_lord.use_reference import verified_reference
    import pytest
    verified_reference()
    read=Path.read_bytes
    for changed in ['core/dif_lord.py','validation/dif_wald_lord/results/dif2pl_mixed/python_fits.npz']:
        with monkeypatch.context() as patch:
            patch.setattr(Path,'read_bytes',lambda p,_changed=changed: b'changed' if str(p)==_changed else read(p))
            with pytest.raises(ValueError,match='identity changed'):verified_reference()
