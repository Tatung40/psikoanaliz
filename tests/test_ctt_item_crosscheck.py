import numpy as np
import pandas as pd
from core.ctt_dual_engine import compare_ctt

def test_three_independent_item_metrics():
    rng=np.random.default_rng(204)
    latent=rng.normal(size=(180,1))
    x=pd.DataFrame((latent+rng.normal(size=(180,8))>0).astype(int),columns=[f"Q{i}" for i in range(8)])
    result=compare_ctt(x,list(x.columns))
    cmp=result["item_crosscheck"]
    assert len(cmp)==8
    assert cmp["p farkı"].max()<.001
    assert cmp["r_ir farkı"].max()<.001
    assert cmp["alfa silme farkı"].max()<.001
