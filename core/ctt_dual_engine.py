"""Independent Cronbach alpha check and conservative CTT item flags."""
import numpy as np
import pandas as pd
from core.reliability import compute_reliability_indices
from core.item_analysis import compute_item_statistics
from core.ctt_item_crosscheck import independent_item_metrics,item_crosscheck
from core.ctt_split_crosscheck import independent_splits,compare_splits

def compare_ctt(data,items):
    x=data[list(items)].apply(pd.to_numeric,errors="coerce")
    clean=x.dropna()
    if len(items)<3 or len(clean)<3:
        raise ValueError("At least three items and respondents required")
    if (clean.nunique()<2).any() or not np.isfinite(clean.to_numpy(float)).all():
        raise ValueError("Constant items or nonfinite values")
    binary=bool(np.isin(clean.to_numpy(float),[0,1]).all())
    native=compute_reliability_indices(clean,list(items),is_dichotomous=binary)
    alpha=float(native["cronbach_alpha"])
    stats,_=compute_item_statistics(clean,list(items),is_dichotomous=binary,overall_alpha=alpha)
    try:
        import pingouin as pg
        other,ci=pg.cronbach_alpha(data=clean,nan_policy="listwise")
        other=float(other)
        ci=tuple(float(y) for y in ci)
        difference=abs(alpha-other)
    except ImportError:
        other=None;ci=None;difference=None
    cross=item_crosscheck(stats,independent_item_metrics(clean))
    split_comparison=compare_splits(native,independent_splits(clean))
    return dict(item_crosscheck=cross,split_comparison=split_comparison,native_alpha=alpha,reference_alpha=other,ci=ci,difference=difference,
                item_stats=stats,N=len(clean),k=len(items),binary=binary,
                removed=len(x)-len(clean))

def item_guidance(result):
    rows=[]
    for _,v in result["item_stats"].iterrows():
        p=float(v["Güçlük (p)"])
        r=float(v["r_ir (Düzeltilmiş)"])
        concerns=[]
        if np.isfinite(r) and r<.2: concerns.append("Low corrected item-total correlation")
        if result["binary"] and np.isfinite(p) and (p<.1 or p>.9):
            concerns.append("Extreme item facility")
        rows.append({"Madde":str(v["Madde"]),"p":p,"r_ir":r,
                     "Durum":"; ".join(concerns) if concerns else "Review routinely",
                     "Öneri":"Check content, key, distractors and target sample; never auto-delete."})
    return pd.DataFrame(rows)
