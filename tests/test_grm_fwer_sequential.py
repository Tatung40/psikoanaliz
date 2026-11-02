import json
from pathlib import Path
import numpy as np
import pytest
from scipy.stats import binom
from validation.grm_fwer_sequential.statistics import cp,condition,summary,TAIL_ALPHA
from validation.grm_fwer_sequential.runner import generate,ordinal_hashes


def rows(n,k,fail=0):
    return [dict(seed=i,impact='minus',valid=i>=fail,decisions={'holm_event':fail<=i<fail+k,'bh_event':fail<=i<fail+k}) for i in range(n)]


def test_simultaneous_cp_endpoints_independently_invert_binomial():
    for n,k in [(500,25),(1000,50),(1500,75),(500,0),(500,500)]:
        lo,hi=cp(k,n)
        if k:assert binom.sf(k-1,n,lo)==pytest.approx(TAIL_ALPHA,abs=1e-12)
        if k<n:assert binom.cdf(k,n,hi)==pytest.approx(TAIL_ALPHA,abs=1e-12)
    assert 12*TAIL_ALPHA==pytest.approx(.05)


def test_failures_never_certify_nominal_or_manufacture_inflation():
    r=condition(rows(500,0,500))
    assert r['operational_fwer']==1 and r['missing_outcome_robust_cp']==[0,1]
    assert not r['classified']
    assert condition(rows(499,0))['decision']=='INCONCLUSIVE'
    assert condition(rows(500,0))['decision']=='SCOPED CONTROL (LIBERAL .075)'
    assert condition(rows(500,100))['decision']=='SCOPED INFLATION'


def test_same_original_responses_and_actual_paired_draws_without_fits():
    hashes,_=ordinal_hashes();seed=2026110100
    xm,gm,tm=generate(seed,'minus',hashes[seed,'minus'])
    xz,gz,tz=generate(seed,'zero',hashes[seed,'zero'])
    assert tm==tz and len(tm)==12
    np.testing.assert_array_equal(gm,gz);np.testing.assert_array_equal(xm[:500],xz[:500])
    assert (xz[500:]>=xm[500:]).all()
    with pytest.raises(ValueError,match='hash'):generate(seed,'zero','bad')


def test_fixed_looks_and_duplicate_rejection():
    p=json.loads(Path('validation/grm_fwer_sequential/protocol.json').read_text())
    assert p['looks_per_condition']==[500,1000,1500] and p['tail_alpha']==TAIL_ALPHA
    with pytest.raises(ValueError,match='Duplicate'):summary([rows(1,0)[0]]*2)
    a=rows(500,0);b=[{**r,'impact':'zero'} for r in a]
    assert summary(a+b)['stop_at_look']
