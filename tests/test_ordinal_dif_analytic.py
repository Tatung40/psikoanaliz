"""Small deterministic mathematical checks; no fits or random data."""
import itertools
import json

import numpy as np
from scipy.special import expit

from validation.ordinal_dif_analytic.counterexample import CACHE,joint_score,probabilities,table


def test_convolution_matches_independent_pattern_enumeration():
    probs=probabilities(np.array([-.7,.6]),np.array([.9,1.2,1.5]),np.array([[-1.,1.],[-.7,.9],[-.6,1.3]]))
    weights=np.array([.4,.6])
    for items in ([0,1,2],[1,2],[1]):
        reference=np.zeros((2*len(items)+1,3))
        for pattern in itertools.product(range(3),repeat=3):
            mass=sum(weights[q]*np.prod([probs[q,i,c] for i,c in enumerate(pattern)]) for q in range(2))
            reference[sum(pattern[i] for i in items),pattern[0]]+=mass
        np.testing.assert_allclose(joint_score(probs,weights,0,items),reference,atol=1e-15)


def test_zero_impact_equal_full_categories_and_latent_model_invariance():
    for score in (list(range(10)),list(range(9))):
        result=table(61,0.,9,score)
        np.testing.assert_allclose(result['category_difference'],0.,atol=1e-14)
        for g in result['groups']:
            np.testing.assert_allclose(np.sum(g['category_probability'],axis=1),1.,atol=1e-14)
    # No group parameter enters p(Y|theta); shifting its distribution is separate.
    p=probabilities(np.array([-.8,.2]),[1.2],[[-1.,0.,1.]])
    np.testing.assert_allclose(p.sum(2),1.)


def test_total_means_reuse_cached_evidence_and_identify_score_category():
    cache=json.loads(CACHE.read_text())
    for j in (8,9):
        item=f'P{j+1:02d}';r=table(81,-.5,j,list(range(10)))
        np.testing.assert_allclose(r['mean_difference'],cache['81_-0.5'][item]['mean_difference'],atol=1e-12)
        peak=r['largest_mean_contrast'];assert peak['score']==(13 if j==8 else 16)
        assert abs(sum(k*d for k,d in enumerate(peak['category_differences']))-peak['difference'])<1e-14


def test_rest_and_invariant_anchor_matching_do_not_guarantee_equality():
    for indices in (list(range(9)),list(range(8))):
        r=table(81,-.5,9,indices)
        assert max(abs(v) for row in r['category_difference'] for v in row)>1e-3
    total=table(61,-.5,9,list(range(10)))
    rest=table(61,-.5,9,list(range(9)))
    assert len(total['mean_difference'])==31 and len(rest['mean_difference'])==28
    assert total['groups'][0]['conditional_mean'][0]==0
    assert rest['groups'][0]['conditional_mean'][0]>0


def test_gh61_gh81_category_stability():
    for indices in (list(range(10)),list(range(9)),list(range(8))):
        a=table(61,-.5,9,indices);b=table(81,-.5,9,indices)
        for x,y in zip(a['groups'],b['groups']):
            np.testing.assert_allclose(x['category_probability'],y['category_probability'],atol=1e-6,rtol=0)


def test_rasch_total_sufficiency_positive_control_does_not_extend_to_rest():
    # At fixed theta, enumerate/convolve a=1 binary Rasch probabilities.
    p=expit(np.array([-1.,1.])[:,None]-np.array([-.8,.1,.7])[None,:])
    probs=np.stack([1-p,p],axis=2)
    def conditional(weights,items):
        joint=joint_score(probs,weights,0,items);return joint/joint.sum(1)[:,None]
    np.testing.assert_allclose(conditional([1,0],[0,1,2]),conditional([0,1],[0,1,2]),atol=1e-14)
    assert np.max(np.abs(conditional([1,0],[1,2])-conditional([0,1],[1,2])))>.1


def test_saved_counterexample_provenance_and_future_precision_bound():
    import hashlib
    import math
    from pathlib import Path
    from validation.ordinal_dif_analytic import counterexample as module
    root=Path(module.__file__).parent
    saved=json.loads((root/'evidence.json').read_text());plan=json.loads((root/'next_experiment_plan.json').read_text())
    assert saved['cache_sha256']==hashlib.sha256(CACHE.read_bytes()).hexdigest()
    assert saved['code_sha256']==hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    assert plan['source_counterexample_sha256']==hashlib.sha256((root/'evidence.json').read_bytes()).hexdigest()
    n=plan['n_required'];target=plan['paired_mcse_target']
    assert 1/math.sqrt(n)<=target<1/math.sqrt(n-1)
    assert plan['primary_ci_halfwidth']>target  # MCSE target is not a 95% CI halfwidth
    assert max(r['cached_mean_max_error'] for r in saved['cached_checks'])<1e-12
    assert 'NOT EXECUTED' in plan['status']
