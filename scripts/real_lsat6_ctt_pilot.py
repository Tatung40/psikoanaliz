"""Real public LSAT6 (Bock & Lieberman) preflight, independent KTK crosscheck.

Dataset stays external; this script does not upload respondent-level rows.
Source: Rdatasets, psych::lsat6; the source dataset is distributed under
its own license. Run manually: python scripts/real_lsat6_ctt_pilot.py
"""
from io import BytesIO
from urllib.request import urlopen
import numpy as np
import pandas as pd
from core.reliability import compute_reliability_indices
from core.ctt_dual_engine import compare_ctt

URL="https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/psych/lsat6.csv"
EXPECTED_N=1000
EXPECTED_P=np.array([.924,.709,.553,.763,.870])
EXPECTED_ALPHA=.2949971922159633

def run():
    with urlopen(URL,timeout=25) as response:
        content=response.read()
    if len(content)>100_000:
        raise AssertionError("Unexpected source download size")
    raw=pd.read_csv(BytesIO(content))
    assert list(raw.columns)==["rownames","Q1","Q2","Q3","Q4","Q5"]
    x=raw[["Q1","Q2","Q3","Q4","Q5"]]
    assert len(x)==EXPECTED_N and x.notna().all().all()
    assert np.isin(x.to_numpy(),[0,1]).all()
    assert np.allclose(x.mean().to_numpy(),EXPECTED_P,atol=1e-12)
    native=compute_reliability_indices(x,list(x.columns),is_dichotomous=True)
    independent=compare_ctt(x,list(x.columns))
    k=x.shape[1]
    v=x.var(ddof=1).sum()
    score_variance=x.sum(axis=1).var(ddof=1)
    alpha=(k/(k-1))*(1-v/score_variance)
    assert abs(alpha-EXPECTED_ALPHA)<1e-12
    assert abs(native["cronbach_alpha"]-alpha)<1e-6
    assert abs(native["kr20"]-alpha)<1e-6
    assert abs(independent["native_alpha"]-alpha)<1e-6
    if independent["reference_alpha"] is None:
        raise AssertionError("Independent Pingouin dependency unavailable")
    assert abs(independent["reference_alpha"]-alpha)<1e-10
    assert abs(native["standardized_alpha"]-(
        k*np.nanmean(np.corrcoef(x,rowvar=False)[~np.eye(k,dtype=bool)])/
        (1+(k-1)*np.nanmean(np.corrcoef(x,rowvar=False)[~np.eye(k,dtype=bool)]))
    ))<1e-6
    print("REAL LSAT6: PASS | N=1000, k=5, CTT alpha=%.9f, KR20=%.9f, Pingouin=%.9f"
          %(alpha,native["kr20"],independent["reference_alpha"]))
    print("Item p:",", ".join(f"{p:.3f}" for p in EXPECTED_P))
    print("Respondent-level records are neither saved nor logged")

if __name__=="__main__":
    run()
