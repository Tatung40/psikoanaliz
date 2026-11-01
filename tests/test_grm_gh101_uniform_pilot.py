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
