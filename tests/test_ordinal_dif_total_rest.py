import math
import numpy as np
import pytest
from scipy.stats import binomtest
from validation.ordinal_dif_total_rest.experiment import paired_stats, summarize, analyze
from validation.ordinal_dif_paired.experiment import classify


def test_discordances_mcnemar_mcse_and_exact_interval():
    x=[1]*8+[0]*12; y=[1]*2+[0]*6+[1]*2+[0]*10
    r=paired_stats(x,y)
    assert (r['a'],r['b'],r['c'],r['d'])==(2,6,2,10)
    assert r['delta']==.2
    assert r['mcnemar_p']==binomtest(6,8,.5).pvalue
    assert r['mcse']==pytest.approx(np.std(np.array(x)-y,ddof=1)/math.sqrt(20))
    assert r['exact_ci'][0]<r['delta']<r['exact_ci'][1]
    assert r['hoeffding_ci'][0]<0<r['hoeffding_ci'][1]


def test_no_discordance_is_not_certainty_and_empty_is_undefined():
    r=paired_stats([0]*30,[0]*30)
    assert r['mcnemar_p']==1 and r['mcse']==0
    assert r['exact_ci'][0]<0<r['exact_ci'][1]
    assert paired_stats([],[])['mcnemar_p'] is None
    with pytest.raises(ValueError): paired_stats([1],[])


def test_failure_never_becomes_no_event_or_confirmatory_success():
    rows=[dict(methods=dict(total=classify([.001,.5],.05),rest=classify([.5,.5],.05))),
          dict(methods=dict(total=classify([.5,.5],.05),rest=classify([np.nan,.5],.05)))]
    r=summarize(rows,2)
    assert r['failures']==1 and not r['fixed_design_complete']
    assert r['paired']['n']==1
    assert r['paired']['unknown_failure_empirical_bounds']==[0.,1.]
    assert r['methods']['rest']['unknown_failure_empirical_bounds']==[0.,.5]
    assert 'INCOMPLETE' in r['primary_decision']


def test_shared_data_and_target_exclusion_only(monkeypatch):
    import pandas as pd
    from validation.ordinal_dif_total_rest import experiment as e
    data=pd.DataFrame({'P09':[0,1,2,1],'P10':[2,1,0,1]});g=np.array([0,0,1,1]);calls=[]
    def fake(y,gf,match,ordinal):
        calls.append((y.copy(),gf.copy(),match.copy(),ordinal))
        return {'p_omnibus':.5}
    monkeypatch.setattr(e,'logistic_dif',fake)
    for method in ('total','rest'): assert analyze(data,g,method)['valid']
    for i in range(2):
        np.testing.assert_array_equal(calls[i][0],calls[i+2][0])
        np.testing.assert_array_equal(calls[i][1],calls[i+2][1])
        np.testing.assert_array_equal(calls[i+2][2],calls[i][2]-calls[i][0])


def test_precision_and_design_preregistered():
    import json
    from validation.ordinal_dif_total_rest.experiment import HERE
    p=json.loads((HERE/'plan.json').read_text())
    assert p['n_required']==2500 and p['feasibility_pilot']==30 and p['budget_seconds']==1800
    assert p['primary_ci_halfwidth']==pytest.approx(math.sqrt(2*math.log(40)/2500))
    assert 'not nominal FWER control' in p['mcnemar']
