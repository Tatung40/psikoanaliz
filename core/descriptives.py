"""Modelden bağımsız veri ön inceleme betimsel istatistikleri."""
import numpy as np
import pandas as pd


def describe_items(frame):
    """Selected numeric item columns; missing values excluded per statistic."""
    x=pd.DataFrame(frame).apply(pd.to_numeric,errors="coerce")
    result=[]
    for name in x:
        v=x[name].dropna()
        n=len(v)
        values=v.to_numpy(dtype=float)
        binary=n>0 and np.isin(values,[0,1]).all()
        result.append({
            "Madde":str(name),
            "Geçerli N":n,
            "Eksik N":int(len(x)-n),
            "Eksik %":100*(len(x)-n)/len(x) if len(x) else np.nan,
            "Ortalama":v.mean() if n else np.nan,
            "Medyan":v.median() if n else np.nan,
            "SS":v.std(ddof=1) if n>=2 else np.nan,
            "Varyans":v.var(ddof=1) if n>=2 else np.nan,
            "Minimum":v.min() if n else np.nan,
            "Maksimum":v.max() if n else np.nan,
            "Çarpıklık":v.skew() if n>=3 else np.nan,
            "Basıklık (fazlalık)":v.kurt() if n>=4 else np.nan,
            "0/1":bool(binary),
            "Doğru yanıt %":100*v.mean() if binary else np.nan,
        })
    return pd.DataFrame(result)


def describe_total(frame):
    """Sum only for complete numeric cases; never treat missing as zero."""
    x=pd.DataFrame(frame).apply(pd.to_numeric,errors="coerce")
    complete=x.dropna()
    if complete.empty:return pd.Series(dtype=float),None
    scores=complete.sum(axis=1)
    n=len(scores)
    return scores,{
        "Geçerli N":n,
        "Eksik veri nedeniyle dışlanan":len(x)-n,
        "Ortalama":scores.mean(),
        "Medyan":scores.median(),
        "SS":scores.std(ddof=1) if n>=2 else np.nan,
        "Varyans":scores.var(ddof=1) if n>=2 else np.nan,
        "Minimum":scores.min(),
        "Maksimum":scores.max(),
        "Çarpıklık":scores.skew() if n>=3 else np.nan,
        "Basıklık (fazlalık)":scores.kurt() if n>=4 else np.nan,
        "Q1":scores.quantile(.25),
        "Q3":scores.quantile(.75),
    }
