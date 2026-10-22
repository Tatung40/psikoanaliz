import pandas as pd
import pytest
from core.irt_b12_reference import compare_parameter_tables

def params(offset=0):
    return pd.DataFrame({"Madde":["M01","M02"],"a":[1+offset,1.5+offset],
        "b":[0+offset,.3+offset],"c":[0.,.2],"d":[1.,.9]})

def test_identical_estimators_and_truth():
    detail,summary=compare_parameter_tables(params(),params(),params())
    assert len(detail)==8
    assert summary["max_abs_difference"].max()==0
    assert summary["rmse_python"].max()==0

def test_alignment_by_item_not_row_order():
    truth=params()
    r=truth.iloc[::-1].reset_index(drop=True)
    detail,summary=compare_parameter_tables(truth,r)
    assert detail.abs_delta.max()==0

def test_reject_mismatch_and_duplicates():
    r=params().rename(index={0:3}).copy()
    r.loc[r.index[0],"Madde"]="OTHER"
    with pytest.raises(ValueError,match="item IDs differ"):
        compare_parameter_tables(params(),r)
    r=params();r.loc[1,"Madde"]="M01"
    with pytest.raises(ValueError,match="unique"):
        compare_parameter_tables(params(),r)

def test_recovery_distinguishes_two_methods():
    py=params(.2);r=params();truth=params()
    detail,summary=compare_parameter_tables(py,r,truth)
    a=summary.set_index("parameter").loc["a"]
    assert a["rmse_python"]==pytest.approx(.2)
    assert a["rmse_r"]==0
