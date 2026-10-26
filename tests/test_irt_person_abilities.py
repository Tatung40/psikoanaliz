"""Regression tests for shared person-level posterior ability output."""
import numpy as np
import pandas as pd
import pytest
from core.irt_person_abilities import person_abilities

@pytest.mark.parametrize("model,categories",[
    ("Rasch",2),("1PL",2),("2PL",2),("3PL",2),("4PL",2),
    ("GRM",4),("PCM",4),("GPCM",4),("NRM",4),
])
def test_person_abilities_all_models(model,categories):
    theta=np.linspace(-4,4,161)
    rows=[]
    for item in ("I1","I2","I3"):
        for t in theta:
            p=np.exp(np.arange(categories)*t/3)
            p=p/p.sum()
            for c in range(categories):
                rows.append({"Madde":item,"Theta":t,"Kategori":str(c),"Olasılık":p[c]})
    x=np.array([[0,0,0],[categories-1]*3,[1,1,1]])
    result=person_abilities(x,pd.DataFrame(rows),model=model,identifiers=["A","B","C"])
    assert list(result["Birey"])==["A","B","C"]
    assert len(result)==3
    assert result["Yetenek (θ)"].iloc[0]<result["Yetenek (θ)"].iloc[1]
    assert np.isfinite(result["SH (θ)"]).all()
    assert (result["SH (θ)"]>0).all()
    assert (result["Model"]==model).all()

def test_person_abilities_rejects_invalid_codes():
    grid=np.linspace(-4,4,161)
    rows=[{"Madde":item,"Theta":t,"Kategori":str(c),"Olasılık":.5}
          for item in ("I1","I2","I3") for t in grid for c in (0,1)]
    with pytest.raises(ValueError,match="Yanıt kodları"):
        person_abilities(np.array([[0,0,2]]),pd.DataFrame(rows),model="Rasch")
