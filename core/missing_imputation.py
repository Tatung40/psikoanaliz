"""Missing-data diagnostics and guarded scikit-learn imputation.
Never infer MCAR/MAR/MNAR from descriptive missingness alone.
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer

METHODS = ("keep", "listwise", "zero", "mode", "median", "knn")

def diagnose(frame, columns):
    x = frame.loc[:, columns].apply(pd.to_numeric, errors="raise")
    n, k = x.shape
    missing = x.isna()
    count = int(missing.to_numpy().sum())
    rates = missing.mean().sort_values(ascending=False)
    binary = all(set(x[c].dropna().unique()).issubset({0, 1}) for c in columns)
    rows_missing = int(missing.any(axis=1).sum())
    high = [str(c) for c in rates[rates > .30].index]
    recommendation = "keep" if binary else ("mode" if count and all(x[c].dropna().nunique() <= 7 for c in columns) else "median")
    return dict(n=n, k=k, missing=count, missing_pct=100*count/(n*k) if n*k else 0,
                rows_missing=rows_missing, complete_rows=n-rows_missing,
                per_item=rates.rename("missing_rate").reset_index().rename(columns={"index":"item"}),
                binary=binary, high_missing=high, recommended=recommendation,
                warning=("MTK: Eksik yanıtları gözlenmiş 0 puanla karıştırmayın. "
                         "Platformun mevcut ikili MTK motoru eksiksiz matris ister; "
                         "eksik veriyi model olabilirliğinde ele almak ayrıca geliştirilmelidir."
                         if binary else "Tek atama varyansı ve belirsizliği olduğundan küçük gösterebilir."))

def apply(frame, columns, method, *, neighbors=5, binary_round=False):
    if method not in METHODS:
        raise ValueError("Desteklenmeyen yöntem")
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("Madde sütunlarını açıkça seçin.")
    out = frame.copy()
    x = out.loc[:, columns].apply(pd.to_numeric, errors="raise")
    if method == "keep":
        return out
    if method == "listwise":
        return out.dropna(subset=columns).copy()
    if method == "zero":
        x2 = x.fillna(0)
    else:
        if x.isna().all(axis=0).any():
            raise ValueError("Tamamı eksik olan maddeye atama yapılamaz.")
        if method in ("mode", "median"):
            strategy = "most_frequent" if method == "mode" else "median"
            x2 = pd.DataFrame(SimpleImputer(strategy=strategy).fit_transform(x),
                              columns=columns, index=x.index)
        elif method == "knn":
            if neighbors < 1:
                raise ValueError("Komşu sayısı pozitif olmalı.")
            x2 = pd.DataFrame(KNNImputer(n_neighbors=neighbors).fit_transform(x),
                              columns=columns, index=x.index)
    binary = all(set(x[c].dropna().unique()).issubset({0, 1}) for c in columns)
    if binary and method in ("median", "knn") and not binary_round:
        raise ValueError("İkili maddede kesirli puan üretilebilir. Açık yuvarlama onayı gerekir.")
    if binary and binary_round:
        x2 = (x2 >= .5).astype(int)
    # Preserve all observed values exactly; only change missing cells.
    out.loc[:, columns] = x.where(x.notna(), x2)
    return out
