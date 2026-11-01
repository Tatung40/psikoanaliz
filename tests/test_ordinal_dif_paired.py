"""Tiny deterministic paired-study checks; no model fits or MC studies."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from validation.ordinal_dif_paired.experiment import (PLAN_PATH,binomial_ci,classify,precision_plan,score_mechanism,summarize)


def test_plan_has_oracle_disjoint_common_family_and_conservative_precision():
    p=json.loads(PLAN_PATH.read_text())
    assert not set(p['anchors'])&set(p['tested_family'])
    assert len(p['anchors'])+len(p['tested_family'])==p['items']==10
    result=precision_plan(p)
    assert result['required_repetitions']>400
    assert result['worst_ci_deviation']<=.02<result['previous_n_deviation']
    assert result['method_alpha']==pytest.approx(.05/3)


def test_holm_is_one_dataset_event_and_partial_families_are_invalid():
    result=classify([.03,.04],.05)
    assert result['raw_rejections']==[1,1]
    assert not result['holm_event'] and result['bh_event']
    assert not classify([.001,np.nan],.05)['valid']
    assert not classify([.001,.1,.2],.05)['valid']


def test_failed_analyses_and_paired_missingness_are_not_zero_imputed():
    p=json.loads(PLAN_PATH.read_text());prec={'required_repetitions':4000,'method_alpha':.05/3}
    records=[{'methods':{m:classify([.001,.5],.05) for m in p['methods']}},
             {'methods':{m:classify([.9,.9],.05) for m in p['methods']}}]
    records[1]['methods']['GRM_LR']={'valid':False,'reason':'timeout'}
    result=summarize(records,p,prec)
    grm=result['methods']['GRM_LR']
    assert grm['failed']==1 and grm['failure_range']==[.5,1.]
    assert grm['bh_fdr']==grm['bh_fwer']==1.
    assert result['paired']['Mantel minus GRM_LR']['complete_pairs']==1
    assert result['paired']['Mantel minus GRM_LR']['missing_pair_range']==[-.5,.5]
    assert result['scientific_status']=='PILOT ONLY / NOT CERTIFIED'
    assert all(r['decision']=='INCONCLUSIVE' for r in result['methods'].values())


def test_known_model_score_convolution_zero_impact_and_normalization():
    result=score_mechanism(31,0.)
    for row in result.values():
        np.testing.assert_allclose(row['mean_difference'],0.,atol=1e-14)
        for group in row['groups']:
            assert sum(group['total_probability'])==pytest.approx(1.)
            assert min(group['conditional_mean'])>=0 and max(group['conditional_mean'])<=3+1e-12


def test_zero_events_are_not_a_zero_width_interval():
    assert binomial_ci(0,10,.05/3)[1]>.05
    assert binomial_ci(0,0)==[0.,1.]
