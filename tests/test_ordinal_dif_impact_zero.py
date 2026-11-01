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


def test_independent_count_and_covariance_audit():
    zero=[row(i,t,r) for i,(t,r) in enumerate([(1,0),(0,1),(0,0),(1,1)])]
    minus=[row(i,t,r) for i,(t,r) in enumerate([(0,1),(0,0),(0,0),(1,1)])]
    for r in zero:
        for method in r['methods'].values():
            method['raw']=[{'p_omnibus':.001 if method['holm_event'] else .5},{'p_omnibus':.8}]
            method['bh_event']=method['holm_event']
    audit=X.audit_results(zero,minus,X.compare(zero,minus))
    assert audit['raw_zero_method_decisions_checked']==8
    assert audit['interaction_from_four_cell_covariance']['mean']==.25
    assert audit['contrasts']['total_impact_change']['b']==1
    assert audit['scientific_certification'] is False


def test_saved_control_evidence_pairing_and_immutable_negative_results():
    import gzip
    output=X.HERE/'results'
    zero=X.load_records(output/'records.jsonl.gz')
    old=X.load_records(X.OLD/'results/records.jsonl.gz')
    old_by_seed={r['seed']:r for r in old}
    summary=json.loads((output/'summary.json').read_text())
    comparison=json.loads((output/'comparison.json').read_text())
    for key,value in X.E.summarize(zero).items():assert summary[key]==value
    assert comparison==X.compare(zero,old)
    meta=json.loads((output/'metadata.json').read_text())
    assert meta['actual_impact']==0 and X.protect()==39
    assert X.digest((X.OLD/'results/records.jsonl.gz').read_bytes())==meta['negative_records_sha256']
    for name,expected in meta['sources'].items():assert X.digest((X.ROOT/name).read_bytes())==expected
    raw=gzip.decompress((output/'crn.jsonl.gz').read_bytes())
    assert X.digest(raw)==meta['crn_sha256']
    assert X.digest((output/'crn.jsonl.gz').read_bytes())==meta['crn_gzip_sha256']
    traces=[json.loads(line) for line in raw.splitlines()]
    assert len(traces)==len(zero)
    for i,(r,trace) in enumerate(zip(zero,traces)):
        assert r['seed']==trace['seed']==2026110100+i
        assert trace['control_hash']==r['data_sha256']
        assert trace['negative_hash']==old_by_seed[r['seed']]['data_sha256']
        assert trace['reference_rows_identical'] and trace['focal_responses_monotone']
        assert trace['reconstructed_negative_exactly']
        assert [(d['method'],d['shape']) for d in trace['draws']]==[('standard_normal',[500])]*2+[('random',[1000])]*10
        if 'matrix_file' in r:
            path=output/r['matrix_file']
            assert X.digest(path.read_bytes())==r['matrix_sha256']
            with np.load(path) as data:
                assert X.digest(data['X'].tobytes()+data['group'].tobytes())==r['data_sha256']
                if i<30:
                    with np.load(X.OLD/'results'/f'data_{i:04d}.npz') as negative:
                        np.testing.assert_array_equal(data['X'][:500],negative['X'][:500])
                        np.testing.assert_array_equal(data['group'],negative['group'])
    payload=gzip.decompress((output/'records.jsonl.gz').read_bytes())
    assert X.digest(payload)==summary['records_sha256']
    assert X.digest((output/'records.jsonl.gz').read_bytes())==summary['records_gzip_sha256']
    audit=json.loads((output/'audit.json').read_text())
    assert audit['complete_quadruples']==comparison['all_four_cells_valid']
    assert audit['scientific_certification'] is False
    assert audit['interaction_from_four_cell_covariance']['mean']==pytest.approx(comparison['interaction']['mean'])
