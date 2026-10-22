import numpy as np
import pandas as pd
from scripts.b12_7_monte_carlo import generate,load_truth,summarize,ITEMS

def truth():
    rows=[]
    for grp in ("Kadin","Erkek"):
        for i,item in enumerate(ITEMS):
            rows.append({"grup":grp,"Madde":item,"a":1. if i<8 else 1.2,
                         "b":0.,"c":.15 if i>=16 else 0.,"d":.9 if i>=24 else 1.})
    return pd.DataFrame(rows)

def test_generation_reproducible_and_group_balance():
    a=generate(200,truth(),123,False);b=generate(200,truth(),123,False)
    pd.testing.assert_frame_equal(a,b)
    assert a.Cinsiyet.value_counts().to_dict()=={"Kadin":100,"Erkek":100}

def test_misspecification_is_separate_scenario():
    a=generate(200,truth(),123,False);b=generate(200,truth(),123,True)
    assert not a[ITEMS].equals(b[ITEMS])

def test_truth_guard(tmp_path):
    t=truth().iloc[:-1]
    f=tmp_path/"bad.csv";t.to_csv(f,index=False)
    import pytest
    with pytest.raises(ValueError,match="Missing"):
        load_truth(f)

def test_summary_tracks_failed_replicates():
    df=pd.DataFrame([{"N":500,"condition":"testlet","status":"ok","converged":True,
                      "boundary_count":2,"Kadin_4PL_a_rmse":.5},
                     {"N":500,"condition":"testlet","status":"error","converged":False}])
    s=summarize(df).iloc[0]
    assert s.convergence_rate==.5 and s.failure_count==1
    assert s.mean_Kadin_4PL_a_rmse==.5
