"""Reproducible dataset-based CTT pilot (repository sample; NOT real research data)."""
from pathlib import Path
import numpy as np
import pandas as pd
from core.ctt_dual_engine import compare_ctt

DATA=Path(__file__).resolve().parents[1]/"sample_data"/"ikili_puanlanmis.csv"

def test_repo_dataset_ctt_all_independent_reference_gates():
    df=pd.read_csv(DATA,encoding="utf-8-sig")
    items=[c for c in df if c.startswith("Item_")]
    assert len(items)==25 and len(df)==300
    result=compare_ctt(df,items)
    assert result["N"]==300 and result["k"]==25 and result["binary"]
    assert result["reference_alpha"] is not None, "Pingouin independent engine missing"
    assert np.isfinite(result["native_alpha"])
    assert result["difference"]<2e-5
    compare=result["item_crosscheck"]
    assert len(compare)==25 and (compare["Karar"]=="UYUMLU").all(), compare.loc[compare["Karar"]!="UYUMLU"].to_string(index=False)
    for col in ("p farkı","r_ir farkı","alfa silme farkı"):
        assert np.isfinite(compare[col]).all(),col
        assert float(compare[col].max())<1e-5,col
    halves=result["split_comparison"]
    assert len(halves)==2 and (halves["Karar"]=="UYUMLU").all(),halves.to_string(index=False)
    assert float(halves[["r farkı","SB farkı"]].max().max())<1e-5
    print("SAMPLE_PILOT KTK N=300 k=25 ALPHA_NATIVE={:.8f} ALPHA_PINGOUIN={:.8f} MAX_ITEM_DIFF={:.3g} MAX_SPLIT_DIFF={:.3g}".format(
        result["native_alpha"],result["reference_alpha"],compare[["p farkı","r_ir farkı","alfa silme farkı"]].to_numpy().max(),
        halves[["r farkı","SB farkı"]].to_numpy().max()))

def test_repo_dataset_ctt_missingness_listwise_is_identical():
    df=pd.read_csv(DATA,encoding="utf-8-sig")
    items=[c for c in df if c.startswith("Item_")]
    df.loc[:4,items[0]]=np.nan
    outcome=compare_ctt(df,items)
    assert outcome["N"]==295 and outcome["removed"]==5
    assert outcome["difference"]<2e-5
    assert (outcome["item_crosscheck"]["Karar"]=="UYUMLU").all()
