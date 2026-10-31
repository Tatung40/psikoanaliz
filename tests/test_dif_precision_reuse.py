"""No estimation: independent counterexamples and saved-record replay."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from validation.dif_validation_study import clustered_rate, cp, main, summarize

SOURCE = Path(__file__).resolve().parents[1]/'validation/evidence/p2_dif/dif_validation_study.json'


def record(seed, rejected=False):
    p = [.001 if rejected else .9]*20
    return dict(kind='binary',method='MH',n_ref=500,n_foc=500,mu=0.,dif=False,seed=seed,
                status='OK',p=p,p_bh=p,p_holm=p,n_untestable=0,seconds=0.)


def test_perfect_within_dataset_dependence_does_not_create_precision():
    parts=[np.full((20,1),v) for v in (1,1,0,0)]
    precision, ci=clustered_rate(parts,20)
    assert precision['coverage_mc_se']==pytest.approx(np.sqrt(1/12))
    doubled, ci2=clustered_rate([np.repeat(p,10,axis=0) for p in parts],200)
    assert doubled['coverage_mc_se']==pytest.approx(precision['coverage_mc_se'])
    assert ci2==pytest.approx(ci)
    # All-zero samples must not produce a zero-width certainty interval.
    assert clustered_rate([np.zeros((20,1))]*100,20)[1][1]>0


def test_null_fdr_equals_fwer_and_failures_are_not_imputed_as_zero():
    records=[record(0,True),record(1),dict(record(2),status='FAILED')]
    row=summarize(records)[0]
    assert row['fdr_bh']==row['fwer_bh']==.5
    assert row['failed']==1 and row['attempted']==3
    assert row['fdr_bh_failure_range']==pytest.approx([1/3,2/3])
    assert row['fwer_bh_ci']==cp(1,2)  # binomial unit is a whole dataset
    assert row['type1_raw_precision']['independent_replications']==2


def test_variable_coverage_denominators_and_capacity_are_respected():
    parts=[np.array([[1],[1]]),np.array([[0]])]
    p, ci=clustered_rate(parts,2)
    assert p['coverage_mc_se']==pytest.approx(4/9)
    assert ci[0]<=2/3<=ci[1]
    with pytest.raises(ValueError):
        clustered_rate(parts,1)


def test_saved_record_replay_preserves_estimates_flags_and_source(tmp_path):
    raw=SOURCE.read_bytes();original=json.loads(raw)
    result=main(tmp_path,reuse=SOURCE)
    assert SOURCE.read_bytes()==raw
    assert result['scientific_status']=='NOT CERTIFIED'
    assert result['n_jobs']==2560 and sum(r['failed'] for r in result['summary'])==1
    assert len(result['summary'])==len(original['summary'])
    for old,new in zip(original['summary'],result['summary']):
        for key,value in old.items():
            if not key.endswith('_ci'):
                assert new[key]==value,(key,old,new)
    row=next(r for r in result['summary'] if r['method']=='IRTLR')
    assert row['coverage_a_precision']['independent_replications']==row['ok']


def test_duplicate_replications_and_source_overwrite_are_rejected(tmp_path):
    original=json.loads(SOURCE.read_bytes())
    bad=copy.deepcopy(original);bad['runs'].append(bad['runs'][0])
    source=tmp_path/'source.json';source.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='Duplicate'):
        main(tmp_path,reuse=source)
    with pytest.raises(ValueError,match='overwrite'):
        main(SOURCE.parent,reuse=SOURCE)
