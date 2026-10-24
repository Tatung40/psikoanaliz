import numpy as np
import pandas as pd
import pytest
from core.missing_imputation import diagnose, apply

def sample():
    return pd.DataFrame({"id":["a","b","c","d"],"I01":[1.,0.,np.nan,1.],
                         "I02":[0.,1.,1.,np.nan],"group":[1,1,2,2]})

def test_diagnostics_and_original_unchanged():
    df=sample()
    d=diagnose(df,["I01","I02"])
    assert d["missing"]==2 and d["binary"] and d["recommended"]=="keep"
    assert d["rows_missing"]==2
    assert df["I01"].isna().sum()==1

@pytest.mark.parametrize("method",["zero","mode","median","knn"])
def test_imputation_only_missing(method):
    df=sample()
    out=apply(df,["I01","I02"],method,binary_round=(method in ("median","knn")))
    assert out[["I01","I02"]].notna().all().all()
    assert out["id"].equals(df["id"]) and out["group"].equals(df["group"])
    assert np.array_equal(out["I01"].iloc[[0,1,3]].to_numpy(),np.array([1.,0.,1.]))
    assert set(out[["I01","I02"]].to_numpy().ravel()).issubset({0,1})

def test_keep_and_listwise():
    df=sample()
    assert apply(df,["I01","I02"],"keep").equals(df)
    assert len(apply(df,["I01","I02"],"listwise"))==2

def test_binary_knn_requires_consent():
    with pytest.raises(ValueError,match="yuvarlama"):
        apply(sample(),["I01","I02"],"knn")

def test_all_missing_column_rejected():
    df=sample()
    df["I03"]=np.nan
    with pytest.raises(ValueError,match="Tamamı eksik"):
        apply(df,["I01","I03"],"mode")
