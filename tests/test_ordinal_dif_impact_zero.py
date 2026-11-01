import hashlib
import json
import math

import numpy as np
import pytest
from validation.dif_validation_study import gen_ordinal
from validation.ordinal_dif_impact_zero import experiment as X


def row(seed,t,r):
    def outcome(v):return {'valid':v is not None,'holm_event':bool(v)}
    return {'seed':seed,'methods':{'total':outcome(t),'rest':outcome(r)}}


def test_interaction_uses_paired_quadruples_not_independent_variances():
    zero=[row(i,t,r) for i,(t,r) in enumerate([(1,0),(0,1),(0,0),(1,1)])]
    minus=[row(i,t,r) for i,(t,r) in enumerate([(0,1),(0,0),(0,0),(1,1)])]
    report=X.compare(zero,minus)
    w=np.array([2,-1,0,0])
    r=report['interaction']
    assert r['mean']==.25
    assert r['mcse']==pytest.approx(w.std(ddof=1)/2)
    assert sum(r['quadruple_table'].values())==4
    assert sum(r['signed_event_table'].values())==4
    assert r['quadruple_table']['1001']==1
    assert 'NOT APPLICABLE' in r['mcnemar']
    assert report['impact_changes']['total']['b']==1
    assert report['impact_changes']['rest']['c']==1
    assert not report['confirmatory_fixed_design_complete']


def test_shared_draws_reconstruct_old_hash_and_keep_reference_rows():
    seed=2026110100
    minus,g=gen_ordinal(10,11,-.5,False,seed)
    old={'data_sha256':X.data_hash(minus,g)}
    zero,g0,trace=X.paired_control(seed,old,10,11)
    np.testing.assert_array_equal(g,g0)
    np.testing.assert_array_equal(zero.iloc[:10],minus.iloc[:10])
    assert (zero.iloc[10:].to_numpy()>=minus.iloc[10:].to_numpy()).all()
    assert len(trace['draws'])==12 and trace['reconstructed_negative_exactly']
    assert trace['negative_hash']==old['data_sha256']
    with pytest.raises(ValueError,match='archived'):
        X.paired_control(seed,{'data_sha256':'incorrect'},10,11)


def test_missing_pairs_are_not_imputed_as_no_event_and_seed_identity_checked():
    zero=[row(1,1,0),row(2,0,None)]
    minus=[row(1,0,1),row(2,0,1)]
    r=X.compare(zero,minus)
    assert r['interaction']['n']==1 and r['all_four_cells_valid']==1
    assert r['interaction']['unknown_failure_empirical_bounds']==[0.,2.]
    assert not r['confirmatory_fixed_design_complete']
    with pytest.raises(ValueError,match='Duplicate'):
        X.compare([zero[0],zero[0]],minus)
    with pytest.raises(ValueError,match='Missing'):
        X.compare([row(3,0,0)],minus)


def test_exact_interval_matches_reused_binary_interval_and_no_certainty_at_zero():
    from validation.ordinal_dif_total_rest.experiment import paired_stats
    r=paired_stats([1,0,0,0],[0,1,1,0])
    assert X.exact_paired_ci(1,2,4)==r['exact_ci']
    lo,hi=X.exact_paired_ci(0,0,30)
    assert lo<0<hi
    simultaneous=X.mean_stats([0]*2500,4,.05/4)['hoeffding_ci']
    assert simultaneous[1]==pytest.approx(math.sqrt(8*math.log(160)/2500))
    assert simultaneous[1]>math.sqrt(8*math.log(40)/2500)


def test_preregistered_scope_and_pr120_protection():
    p=json.loads((X.HERE/'plan.json').read_text())
    assert p['impact']==0 and p['n_required']==2500 and p['feasibility_pilot']==30
    assert p['budget_seconds']==1800 and 'FOUR contrasts' in p['multiplicity_scope']
    assert X.protect()==39
    assert 'not' in p['causal_scope']
