"""Preflight explicitly separates pre- and post-fit IRT assumptions."""
import numpy as np
import pandas as pd
from core.irt_assumptions import preflight

def data(n=110,k=6):
    r=np.random.default_rng(49)
    return pd.DataFrame(r.binomial(1,.45,size=(n,k)),columns=[f"M{i}" for i in range(k)])

def test_good_binary_does_not_claim_local_independence_passed():
    r=preflight(data(),"binary")
    assert not r["blocked"]
    loc=r["table"].loc[r["table"]["Varsayım / kontrol"]=="Yerel bağımsızlık"]
    assert loc.iloc[0]["Durum"]=="MODEL SONRASI"
    assert "kanıtlandığı" in r["explanation"]

def test_missing_and_invalid_block_fit():
    x=data().astype(float);x.iloc[0,0]=np.nan
    assert preflight(x,"binary")["blocked"]
    x=data();x.iloc[0,0]=4
    assert preflight(x,"binary")["blocked"]

def test_small_sample_warning_and_constant_block():
    x=data(n=55)
    assert not preflight(x,"binary")["blocked"]
    assert "UYARI" in set(preflight(x,"binary")["table"]["Durum"])
    x["M1"]=0
    assert preflight(x,"binary")["blocked"]

def test_nominal_codes_not_assumed_ordinal():
    x=pd.DataFrame((np.arange(100)[:,None]+np.arange(5)[None,:])%5)
    r=preflight(x,"nominal")
    assert not r["blocked"]
    assert any("Nominal NRM" in t for t in r["table"]["Açıklama"])

def test_screen_navigation():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    at=AppTest.from_file(str(Path(__file__).resolve().parents[1]/"app.py"),default_timeout=120)
    x=data()
    for key,val in {"raw_df":x,"scored_df":None,"item_cols":list(x.columns),"id_col":None,
                    "group_col":None,"nav_page":"MTK: Varsayım Denetimi"}.items():
        at.session_state[key]=val
    at.run()
    assert not at.exception
    assert any(b.key=="irt_check_run" for b in at.button)
    at.button(key="irt_check_run").click().run()
    assert not at.exception
    assert "irt_assumption_last" in at.session_state
