import numpy as np
import pandas as pd
from pathlib import Path
from core.scoring import score_dichotomous, calculate_total_scores

ROOT=Path(__file__).resolve().parents[1]
ITEMS=[f"M{i:02d}" for i in range(1,21)]
KEY="BBDACDBBDAACABBBBACA"

def test_real_ktk_reference():
    df=pd.read_csv(ROOT/"tests/fixtures/real_ktk_30x20.csv")
    x=df[ITEMS].to_numpy(dtype=float)
    assert x.shape==(30,20)
    assert not np.isnan(x).any()
    total=x.sum(axis=1)
    np.testing.assert_array_equal(total,df["Toplam"].to_numpy())
    np.testing.assert_allclose(total.mean(),13.2,atol=1e-10)
    np.testing.assert_allclose(total.std(ddof=1),3.6236769285849175,atol=1e-10)
    assert (total.min(),total.max())==(6,19)
    np.testing.assert_allclose(x.mean(),0.66,atol=1e-12)
    alpha=20/19*(1-x.var(axis=0,ddof=1).sum()/total.var(ddof=1))
    np.testing.assert_allclose(alpha,0.7411912133274361,atol=1e-12)
    np.testing.assert_allclose(total.std(ddof=1)*np.sqrt(1-alpha),1.843482330329845,atol=1e-10)
    assert np.corrcoef(x[:,6],total-x[:,6])[0,1]<0
    assert np.corrcoef(x[:,10],total-x[:,10])[0,1]<0
    raw=pd.DataFrame({"Ogrenci":df["Ogrenci"]})
    for j,item in enumerate(ITEMS):
        wrong=next(c for c in "ABCD" if c!=KEY[j])
        raw[item]=np.where(x[:,j]==1,KEY[j],wrong)
    scored,_=score_dichotomous(raw,ITEMS,dict(zip(ITEMS,KEY)),id_column="Ogrenci")
    np.testing.assert_array_equal(scored[ITEMS].to_numpy(),x)
    np.testing.assert_array_equal(calculate_total_scores(scored,ITEMS).to_numpy(),total)
