"""Independent split-half reliability checks on the exact listwise analysis sample."""
import numpy as np
import pandas as pd

def independent_splits(data):
    x=pd.DataFrame(data).astype(float)
    if x.ndim!=2 or len(x)<3 or x.shape[1]<3 or not np.isfinite(x.to_numpy()).all():
        raise ValueError("At least three persons/items and finite complete values required")
    names=list(x.columns)
    def half(left,right,label):
        a=x[left].sum(axis=1).to_numpy()
        b=x[right].sum(axis=1).to_numpy()
        if np.std(a)==0 or np.std(b)==0:
            return {"Bölme":label,"Yarılar arası r":np.nan,"Spearman-Brown":np.nan}
        r=float(np.corrcoef(a,b)[0,1])
        sb=2*r/(1+r) if r>-1+1e-12 else np.nan
        return {"Bölme":label,"Yarılar arası r":r,"Spearman-Brown":sb}
    return pd.DataFrame([
        half(names[::2],names[1::2],"Tek-çift"),
        half(names[:len(names)//2],names[len(names)//2:],"İlk-son")])

def compare_splits(native,splits):
    fields=(("Tek-çift","split_half_odd_even_r","spearman_brown_odd_even"),
            ("İlk-son","split_half_first_second_r","spearman_brown_first_second"))
    rows=[]
    for label,r_key,sb_key in fields:
        other=splits.loc[splits["Bölme"]==label].iloc[0]
        r_native=float(native.get(r_key,np.nan))
        sb_native=float(native.get(sb_key,np.nan))
        r_diff=abs(r_native-other["Yarılar arası r"])
        sb_diff=abs(sb_native-other["Spearman-Brown"])
        rows.append({"Bölme":label,"Yerel r":r_native,"Bağımsız r":other["Yarılar arası r"],
          "r farkı":r_diff,"Yerel Spearman-Brown":sb_native,
          "Bağımsız Spearman-Brown":other["Spearman-Brown"],"SB farkı":sb_diff,
          "Karar":"UYUMLU" if np.isfinite(r_diff) and np.isfinite(sb_diff) and max(r_diff,sb_diff)<=.001 else "İNCELENMELİ"})
    return pd.DataFrame(rows)
