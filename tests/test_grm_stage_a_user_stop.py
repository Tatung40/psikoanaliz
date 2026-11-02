"""Resource-stop accounting and cached hashes only; never calibrates models."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import pytest
from tests.test_grm_gh101_stage_a import fixture
from validation.grm_gh101_stage_a import runner as R
from validation.grm_gh101_stage_a import statistics as S
from validation.grm_gh101_stage_a import user_stop_finalize as F


def subset():
    rows,ordinal=fixture()
    masked={key:value if key[0] in F.SEEDS else dict(methods={m:dict(valid=False) for m in ('total','rest')}) for key,value in ordinal.items()}
    return rows[:60],masked


def test_30_denominator_preserves_statistical_values_and_original_plan():
    rows,ordinal=subset();base=S.summarize(rows,ordinal);snapshot=copy.deepcopy(base)
    result=F.retarget30(base);audit=F.verify30(result,rows,ordinal,base)
    assert base==snapshot and base['planned_analyses']==200
    assert result['planned_analyses']==60 and result['not_run']==0 and audit['valid_pairs']==30
    for impact in ('minus','zero'):
        before,after=base['conditions'][impact]['holm_fwer'],result['conditions'][impact]['holm_fwer']
        for field in ('rate','mcse','cp95','cp975','joint11_cp'):assert before[field]==after[field]
        assert after['unknown_failure_empirical_bounds']==[after['rate'],after['rate']]
    assert all(v['n']==30 and v['missing_from_fixed30']==0 for v in result['contrasts'].values())


def test_invalid_and_incomplete_pairs_remain_missing():
    rows,ordinal=subset();rows=rows[:57]
    rows[0].update(valid=False,decisions=S.decisions({'valid':False}))
    base=S.summarize(rows,ordinal);result=F.retarget30(base);audit=F.verify30(result,rows,ordinal,base)
    assert result['attempted']==57 and result['invalid']==1 and result['not_run']==3
    assert audit['valid_pairs']==27
    assert result['conditions']['minus']['holm_fwer']['n']==28
    assert result['contrasts']['GRM_zero-minus-minus']['n']==27


def test_original_runner_cannot_repeat_existing_evidence(monkeypatch):
    if not (R.HERE/'results/metadata.json').exists():pytest.skip('Evidence not yet committed')
    monkeypatch.setattr(R.G.Fitter,'fit',lambda *a,**kw:pytest.fail('No repeated fits'))
    with pytest.raises(RuntimeError,match='Evidence exists'):R.run()


def test_cached_60_boundary_and_sha256():
    out=R.HERE/'results';snapshot=R.HERE/'user_stop_snapshot'
    if not (out/'resource30_summary.json').exists():pytest.skip('User-stop evidence not yet committed')
    for name,digest in json.loads((out/'SHA256.json').read_text()).items():
        assert hashlib.sha256((out/name).read_bytes()).hexdigest()==digest
    raw=gzip.decompress((out/'records.jsonl.gz').read_bytes())
    prefix=gzip.decompress((snapshot/'records.prefix.jsonl.gz').read_bytes())
    assert raw.startswith(prefix) and len(raw.splitlines())==60
    before=json.loads((snapshot/'SHA256.json').read_text())
    for name,digest in before.items():
        value=prefix if name=='records.prefix.jsonl' else (snapshot/name).read_bytes()
        assert hashlib.sha256(value).hexdigest()==digest
    boundary=json.loads((snapshot/'boundary.json').read_text());assert boundary['completed']==60
    result=json.loads((out/'resource30_summary.json').read_text())
    assert result['attempted']==60 and result['excess_completed']==0
    assert result['original_stage_status']=='USER-STOPPED / INCOMPLETE'
    assert result['statistical_status']=='FWER NOT CERTIFIED'
    meta=json.loads((out/'metadata.json').read_text())
    assert hashlib.sha256((R.HERE/'protocol.json').read_bytes()).hexdigest()==meta['protocol_sha256']
    audit=json.loads((out/'audit.json').read_text())
    assert audit['rows']==60 and audit['optimizations']==0 and audit['statistical_audit']
