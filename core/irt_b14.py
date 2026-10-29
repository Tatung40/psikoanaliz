"""B14: categorical-response import and conservative model selection.

Ordinal scores (0,1,2,...) and nominal choices (A,B,C,D,E) are different
measurement designs. Keep raw responses and answer keys distinct.
"""
import numpy as np
import pandas as pd

ORDINAL=("GRM","PCM","GPCM","RSM")
NOMINAL=("NRM",)
MODELS=ORDINAL+NOMINAL

def prepare_polytomous(data,model,*,categories=None):
    if model not in MODELS:raise ValueError("Desteklenmeyen çok kategorili MTK modeli.")
    if not isinstance(data,pd.DataFrame):data=pd.DataFrame(data)
    if data.shape[0]<30 or data.shape[1]<3:
        raise ValueError("Pilot için en az 30 katılımcı ve üç madde gerekli.")
    if data.isna().any().any():
        raise ValueError("Eksik yanıt var. Doğrulanmış bir eksik veri yöntemi olmadan kestirim yapılmaz.")
    labels=tuple(str(v).strip().upper() for v in (categories or ("A","B","C","D","E")))
    if model=="NRM":
        if len(set(labels))!=len(labels) or not 3<=len(labels)<=20:
            raise ValueError("Nominal kategori sayısı 3–20 ve etiketleri benzersiz olmalı.")
        code={name:i for i,name in enumerate(labels)}
        values=data.astype(str).apply(lambda col:col.str.strip().str.upper())
        if not values.isin(list(code)).all().all():
            raise ValueError("Yanıtlarda tanımlı olmayan nominal kategoriler var.")
        x=values.replace(code).astype(int)
    else:
        if categories is not None:
            raise ValueError("Sıralı modellerde kategori etiketleri yerine 0,1,... puanları kullanın.")
        x=data.apply(pd.to_numeric,errors="coerce")
        if x.isna().any().any() or not np.isfinite(x.to_numpy(float)).all():
            raise ValueError("Sıralı kategoriler sayısal ve eksiksiz olmalı.")
        raw=x.to_numpy(float)
        if not np.array_equal(raw,np.floor(raw)) or (raw<0).any():
            raise ValueError("Sıralı yanıtlar negatif olmayan tamsayı olmalı.")
        origin=int(raw.min());maximum=int(raw.max())
        if origin not in (0,1):
            raise ValueError("Sıralı kategoriler 0 veya 1'den başlamalı.")
        if not 3<=maximum-origin+1<=20:
            raise ValueError("Sıralı modeller 3–20 yanıt kategorisi gerektirir.")
        labels=tuple(str(i) for i in range(origin,maximum+1))
        x=(x-origin).astype(int)
    expected=set(range(len(labels)))
    observed=set(np.unique(x.to_numpy()))
    if observed!=expected:
        raise ValueError(f"Tüm kategoriler pilot örneklemde bulunmalı. Eksik: {sorted(expected-observed)}")
    if (x.nunique()<2).any():
        raise ValueError("Sabit yanıtlı madde bulundu.")
    return x,labels

def fit_b14(data,model,*,categories=None):
    from core.irt_python_mirt_adapter import fit_python_mirt
    x,labels=prepare_polytomous(data,model,categories=categories)
    fitted=fit_python_mirt(x.to_numpy(),model=model,n_categories=len(labels))
    return {"model":model,"engine":"Python mirt 1.2.0",
            "status":("R mirt 1.48 ile sayısal uyum (tek veri seti) ve Monte Carlo geri kazanımı mevcut; "
                      "genel sertifika değildir (validation/r_mirt_polytomous)"),
            "labels":labels,"items":tuple(map(str,x.columns)),"N":len(x),
            "k":x.shape[1],"result":fitted.fit,"coded":x}
