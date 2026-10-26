"""Experimental B14 model diagnostics using mirt 1.2.0 stable public functions.

No made-up S-X2: failures produce an explicit unavailable record.
"""
import numpy as np
import pandas as pd

def _frame(value):
    if isinstance(value,pd.DataFrame):
        return value.reset_index(drop=False)
    if isinstance(value,dict):
        try:return pd.DataFrame(value)
        except (ValueError,TypeError):return pd.DataFrame([value])
    if hasattr(value,"to_pandas"):
        return value.to_pandas()
    raise ValueError("Diagnostic output cannot be interpreted reliably.")

def model_diagnostics(result):
    if result.get("model")=="NRM":
        return pd.DataFrame([{"Tanı":"Nominal madde uyumu","Durum":"Doğrulanmış uygun API yok",
            "Not":"Infit/Outfit ve toplam puan temelli S-X² nominal kategori kodlarına uygulanmaz; değer uydurulmaz."}]),{}
    import mirt
    responses=np.asarray(result["coded"],dtype=int)
    fit=result["result"]
    records=[]
    tables={}
    operations=(
        ("Madde uyumu",lambda:mirt.itemfit(fit,responses,statistics=["infit","outfit","S_X2"])),
        ("Model uyumu",lambda:mirt.compute_fit_indices(fit.model,responses)),
    )
    for name,call in operations:
        try:
            frame=_frame(call())
            if frame.empty:raise ValueError("No results returned")
            if frame.select_dtypes(include="number").replace([np.inf,-np.inf],np.nan).isna().all().all() and not frame.select_dtypes(include="number").empty:
                raise ValueError("No finite numeric estimates")
            tables[name]=frame
            records.append({"Tanı":name,"Durum":"Motor çıktısı mevcut",
                            "Not":"R mirt ya da bağımsız referansla sayısal doğrulama bekleniyor."})
        except Exception as exc:
            records.append({"Tanı":name,"Durum":"Hesaplanamadı",
                            "Not":f"{type(exc).__name__}: {str(exc)[:180]}"})
    return pd.DataFrame(records),tables
