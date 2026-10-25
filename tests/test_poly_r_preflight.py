"""No calibrations: probe existing models and test preflight failure reporting."""
import json
import mirt
from validation import poly_r_preflight as P


def test_missing_r_records_not_executed_and_never_fits(tmp_path,monkeypatch):
    monkeypatch.setattr(mirt,'fit_mirt',lambda *a,**k:(_ for _ in ()).throw(AssertionError('fit forbidden')))
    monkeypatch.setattr(P.shutil,'which',lambda _:None)
    r=P.run(tmp_path)
    assert r['status']==r['R_execution']=='NOT EXECUTED'
    assert r['fits']==r['simulations']==0 and r['scientific_pass'] is False
    assert set(r['python_checks'])=={'GRM','PCM'}
    assert r['GH']['estimator_integration']=='NOT EXECUTED'
    assert (tmp_path/'integration.csv').exists()
    pcm=P.pd.read_csv(tmp_path/'PCM_physical.csv')
    assert (pcm.a==1).all() and pcm.loc[0,'b1']>pcm.loc[0,'b2']
    assert json.loads((tmp_path/'report.json').read_text())['R_checks']=='PENDING'


def test_r_error_is_preserved_and_never_becomes_success(tmp_path,monkeypatch):
    from types import SimpleNamespace
    P.helpers()  # Reuse pinned Git helper before mocking only the R process.
    monkeypatch.setattr(P.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1,stdout='probe log',stderr='layout mismatch'))
    r=P.run(tmp_path,rscript='mock-Rscript')
    assert r['status']=='FAIL' and r['R_execution']=='EXECUTED' and not r['scientific_pass']
    assert (tmp_path/'R_stderr.log').read_text()=='layout mismatch'
