import json
import math
import numpy as np
import pytest
from validation.grm_lr_full_mc import experiment as E
from validation.grm_lr_full_mc import audit as A


def test_preregistered_budget_stages_and_immutable_pilot():
    p=json.loads((E.HERE/'plan.json').read_text())
    assert [s['pairs'] for s in p['stages']]==[100,500,2500]
    assert p['budget_seconds']==18000 and p['per_analysis_timeout_seconds']==90
    assert p['planned_grm_analyses']==5000 and p['new_maximum_grm_analyses']==4980
    assert E.protect()==111
    pilot=E.pilot_records()
    assert len(pilot)==20 and len({(r['seed'],r['impact']) for r in pilot})==20
    assert all(r['result']['valid'] for r in pilot)
    assert {r['seed'] for r in pilot}==set(range(2026110100,2026110110))


def test_append_only_ledger_seals_without_loss_or_duplicate(tmp_path):
    records=[dict(seed=i,result={'valid':True}) for i in range(103)]
    ledger=E.Ledger(tmp_path)
    for r in records:ledger.append(r)
    assert len(list(tmp_path.glob('chunk_*.jsonl.gz')))==1
    assert list(E.stream_new(tmp_path))==records
    ledger.seal()
    assert not (tmp_path/'active.jsonl').exists()
    assert list(E.stream_new(tmp_path))==records


def test_independent_intervals_and_paired_counts():
    a=.05/9
    assert A.cp(0,10,a)[1]==pytest.approx(1-(a/2)**.1)
    assert A.cp(10,10,a)[0]==pytest.approx((a/2)**.1)
    r=A.binary([1,0,0,0],[0,1,1,0])
    assert (r['b'],r['c'])==(1,2) and r['delta']==-.25
    assert r['mcse']==pytest.approx(np.std([1,-1,-1,0],ddof=1)/2)
    assert r['mcnemar_p']==1.
    lo,hi=A.pair_ci(0,0,10)
    assert lo<0<hi
    assert A.binary([],[])['mcnemar_p'] is None


def test_raw_item_holm_bh_reconstruction():
    h,b=A.holm_bh([.04,.03])
    np.testing.assert_allclose(h,[.06,.06])
    np.testing.assert_allclose(b,[.04,.04])
    assert not np.any(h<.05) and np.any(b<.05)
    with pytest.raises(AssertionError):A.holm_bh([.1,float('nan')])


def test_saved_staged_evidence_and_independent_audit_without_fits():
    output=E.HERE/'results'
    saved=json.loads((output/'summary.json').read_text());metadata=json.loads((output/'metadata.json').read_text())
    records=list(E.stream_new(output));pilot=E.pilot_records()
    assert len(records)==saved['new_attempted']
    keys=[(r['seed'],r['impact']) for r in pilot+records]
    assert len(keys)==len(set(keys))==saved['attempted']
    assert saved['pilot_reused_analyses']==20
    assert all(2026110100<=r['seed']<=2026112599 for r in records)
    assert all(r['seed']>=2026110110 for r in records)
    assert E.protect()==111
    assert E.Z.digest((E.PILOT/'records.json.gz').read_bytes())==metadata['pilot_records_sha256']
    for path,digest in metadata['sources'].items():assert E.Z.digest((E.ROOT/path).read_bytes())==digest
    archived=json.loads((output/'audit.json').read_text());recomputed=A.audit(output)
    def compare(a,b):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:compare(a[k],b[k])
        elif isinstance(a,list):
            assert len(a)==len(b)
            for x,y in zip(a,b):compare(x,y)
        elif isinstance(a,float):assert a==pytest.approx(b,rel=1e-10,abs=1e-12)
        else:assert a==b
    compare(archived,recomputed)
    assert archived['scientific_certification'] is False
    for stage in (100,500,2500):
        path=output/f'stage_{stage:04d}.json'
        if path.exists():
            boundary=json.loads(path.read_text());assert boundary['analyses']==boundary['valid']==2*stage
    for r in records:
        if 'failed_matrix' in r:
            path=output/r['failed_matrix'];assert E.Z.digest(path.read_bytes())==r['matrix_sha256']
            with np.load(path) as data:assert E.Z.digest(data['X'].tobytes()+data['group'].tobytes())==r['data_sha256']
