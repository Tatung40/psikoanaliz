import numpy as np
import pandas as pd
from core.descriptives import describe_items, describe_total


def test_describe_items_missing_and_binary():
    x=pd.DataFrame({"M1":[0,1,1,np.nan],"M2":[1,1,0,0]})
    out=describe_items(x)
    assert out.loc[0,"Geçerli N"]==3
    assert out.loc[0,"Eksik N"]==1
    assert np.isclose(out.loc[0,"Doğru yanıt %"],200/3)
    assert out.loc[1,"Geçerli N"]==4
    assert out.loc[0,"0/1"]


def test_describe_total_excludes_incomplete_cases():
    x=pd.DataFrame({"M1":[0,1,np.nan,1],"M2":[1,1,1,0]})
    scores,summary=describe_total(x)
    assert len(scores)==3
    assert summary["Eksik veri nedeniyle dışlanan"]==1
    assert np.isclose(summary["Ortalama"],4/3)


def test_non_numeric_does_not_get_false_numeric_summary():
    x=pd.DataFrame({"Yanıt":["A","B","A"],"M1":[0,1,1]})
    out=describe_items(x)
    assert out.loc[0,"Geçerli N"]==0
    assert np.isnan(out.loc[0,"Ortalama"])
    assert out.loc[1,"Geçerli N"]==3
