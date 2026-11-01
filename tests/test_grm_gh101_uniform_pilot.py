import json
import numpy as np
import pytest
from core import grm_dif as G
from validation.grm_gh101_uniform_pilot import pilot as P
from validation.grm_gh101_uniform_pilot import audit as A


def test_preselected_seed_pairing_and_immutable_archive():
    protocol=json.loads((P.HERE/'protocol.json').read_text())
    assert protocol['seeds']==list(P.SEEDS) and 2026110981 not in P.SEEDS
    assert P.protect()==159
    old=P.legacy_records()
    assert len(old)==20
    for seed in P.SEEDS:
        proof=P.verify_pair(seed,old)
        assert proof['rng_hashes_verified']==12 and proof['reference_rows_identical'] and proof['groups_identical']


def test_every_native_hypothesis_reaches_verified_cold_start_without_calibration(monkeypatch):
    old=P.legacy_records();x,g=P.load_matrix(old[(P.SEEDS[0],'minus')])
    import pandas as pd
    inputs=G.validate_inputs(pd.DataFrame(x,columns=[f'P{i:02d}' for i in range(1,11)]),g,0,1,[f'P{i:02d}' for i in range(1,9)],categories=4)
    fitter=G.Fitter(inputs,points=101);fitter.audit_grid=P.D.reference_grid(161)
    designs,slots=P.layouts();calls=[]
    class Checked(Exception):pass
    def forbidden(fun,x0,*args,**kwargs):
        calls.append(np.array(x0));assert kwargs['method']=='L-BFGS-B'
        assert kwargs['options']==dict(maxiter=2000,ftol=1e-14,gtol=1e-10)
        raise Checked()
    monkeypatch.setattr(G,'minimize',forbidden)
    for key,m in designs.items():
        expected=A.independent_cold_start(m['design'],x)
        np.testing.assert_array_equal(P.cold_start(m['design'],inputs)[0],expected)
        with pytest.raises(Checked):fitter.fit(m['design'].roles,start_fit=None,information=True,max_iter=2000)
        np.testing.assert_array_equal(calls[-1],expected)
        assert calls[-1][-2:].tolist()==[0.,0.]
    assert len(calls)==len(designs)==7 and len(slots)==8


def test_pass_requires_complete_unique_preselected_datasets_and_all_gates():
    rows=[dict(seed=s,impact=k,valid=True) for s in P.SEEDS for k in ('minus','zero')]
    assert P.decision(rows,'fixed pilot complete')=='PASS'
    assert P.decision(rows[:-1],'fixed pilot complete')=='FAIL'
    assert P.decision(rows[:-1]+[rows[0]],'fixed pilot complete')=='FAIL'
    rows[-1]['valid']=False
    assert P.decision(rows,'fixed pilot complete')=='FAIL'


def test_saved_uniform_pilot_and_independent_audit_no_calibration():
    import gzip
    output=P.HERE/'results';raw=gzip.decompress((output/'records.jsonl.gz').read_bytes())
    rows=[json.loads(line) for line in raw.splitlines()];summary=json.loads((output/'summary.json').read_text());meta=json.loads((output/'metadata.json').read_text())
    assert P.D.sha(raw)==summary['records_sha256']
    assert P.D.sha((output/'records.jsonl.gz').read_bytes())==summary['records_gzip_sha256']
    assert P.protect()==159
    assert summary['technical_decision']==P.decision(rows,summary['stop'])
    assert summary['main_stages_started'] is False and summary['p_values_generated'] is False
    assert summary['old_mc_pooled'] is False and summary['data_generated']==0
    for path,digest in meta['sources'].items():assert P.D.sha((P.ROOT/path).read_bytes())==digest
    prior=P.legacy_records();designs,slots=P.layouts()
    for r in rows:
        assert r['data_sha256']==prior[(r['seed'],r['impact'])]['data_sha256']
        x,g=P.load_matrix(prior[(r['seed'],r['impact'])])
        for call in r.get('optimizer_calls',[]):
            expected=A.independent_cold_start(designs[call['solution']]['design'],x)
            np.testing.assert_array_equal(call['x0'],expected)
        if r['valid']:
            assert len(r['optimizer_calls'])==len(r['solutions'])==r['native_optimizer_calls']==7
            assert all(s['passed'] for s in r['solutions'].values())
            assert all(lr['valid'] for lr in r['lr'])
    original=json.loads((output/'audit.json').read_text());recomputed=A.audit(output)
    # Timing is an observation, not a reproducible scientific value.
    original.pop('seconds');recomputed.pop('seconds')
    assert original.keys()==recomputed.keys()
    for key,value in original.items():
        if isinstance(value,float):assert value==pytest.approx(recomputed[key],abs=1e-12)
        else:assert value==recomputed[key]
    assert original['scientific_certification'] is False
