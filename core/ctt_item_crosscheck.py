"""Independent arithmetic checks for item facility, corrected correlation, alpha if deleted."""
import numpy as np
import pandas as pd

def independent_item_metrics(data):
    x=pd.DataFrame(data).astype(float)
    if x.isna().any().any() or not np.isfinite(x.to_numpy()).all():
        raise ValueError("Complete finite scores required")
    k=x.shape[1]
    if k<3 or len(x)<3:raise ValueError("At least three items and persons")
    out=[]
    for name in x:
        item=x[name].to_numpy()
        rest=x.drop(columns=name).sum(axis=1).to_numpy()
        p=float(item.mean())
        corr=float(np.corrcoef(item,rest)[0,1]) if np.std(item)>0 and np.std(rest)>0 else np.nan
        other=x.drop(columns=name)
        k2=other.shape[1]
        total_var=other.sum(axis=1).var(ddof=1)
        alpha_del=float((k2/(k2-1))*(1-other.var(ddof=1).sum()/total_var)) if total_var>0 else np.nan
        out.append({"Madde":str(name),"Bağımsız ortalama":p,
                    "Bağımsız r_ir":corr,"Bağımsız silinirse alfa":alpha_del})
    return pd.DataFrame(out)

def item_crosscheck(native_items,reference):
    keys=["Madde","Güçlük (p)","r_ir (Düzeltilmiş)","Silindiğinde Alfa"]
    native=native_items[keys].copy()
    native["Madde"]=native["Madde"].astype(str)
    merged=native.merge(reference,on="Madde",validate="one_to_one")
    for a,b,label in [
        ("Güçlük (p)","Bağımsız ortalama","p farkı"),
        ("r_ir (Düzeltilmiş)","Bağımsız r_ir","r_ir farkı"),
        ("Silindiğinde Alfa","Bağımsız silinirse alfa","alfa silme farkı")]:
        merged[label]=(pd.to_numeric(merged[a],errors="coerce")-merged[b]).abs()
    merged["Karar"]=np.where(merged[["p farkı","r_ir farkı","alfa silme farkı"]].max(axis=1)>0.001,
                             "YÖNTEM/FARK İNCELENMELİ","UYUMLU")
    return merged
