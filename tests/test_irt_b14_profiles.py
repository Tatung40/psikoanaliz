import numpy as np
import pandas as pd
from core.irt_b14_profiles import empirical_profiles

def sample():
    rng=np.random.default_rng(42)
    return pd.DataFrame(rng.integers(0,4,size=(120,6)),columns=list("ABCDEF"))

def test_ordinal_profiles_sum_to_one():
    p=empirical_profiles(sample(),tuple(map(str,range(4))),"GRM")
    assert not p.empty
    sums=p.groupby(["Madde","Grup"])["Oran"].sum()
    assert np.allclose(sums.to_numpy(),1)

def test_nominal_profiles_do_not_use_arbitrary_order_as_rest_score():
    p=empirical_profiles(sample(),tuple("WXYZ"),"NRM")
    assert set(p["Grup"])=={"Tüm katılımcılar"}
    assert np.allclose(p.groupby("Madde")["Oran"].sum(),1)
