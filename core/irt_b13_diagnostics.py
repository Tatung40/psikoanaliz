"""B13 diagnostics using existing tested estimators; refuse unsafe inference."""
import numpy as np
import pandas as pd

def calculate_diagnostics(x,fit,items,model):
    result={"item_fit":None,"q3":None,"se":None,"messages":[]}
    if model=="Rasch":
        result["messages"].append(
            "Rasch infit/outfit are available in the Rasch native analysis; "
            "separate S-X2 is not reported under free latent SD.")
        return result
    from core.irt_q3 import q3_analysis
    from core.irt_itemfit import sx2_mirt_compatible
    from core.irt_hessian import observed_information_se
    a,b,c,d=(np.asarray(items[p],float) for p in ("a","b","c","d"))
    try:
        q=q3_analysis(x,a,b,model,person_method="WLE",
            converged=bool(fit["converged"]),
            c=c if model in ("3PL","4PL") else None,
            d=d if model=="4PL" else None)
        result["q3"]=q.get("pairs")
    except Exception as exc:
        result["messages"].append("Q3 hesaplanamadı: "+str(exc))
    if not fit["converged"]:
        result["messages"].append(
            "Yakınsama yok: S-X2, gözlenen bilgi SH ve Wald güven aralıkları bastırıldı.")
        return result
    try:
        result["item_fit"]=sx2_mirt_compatible(x,items,model)
    except Exception as exc:
        result["messages"].append("S-X2 hesaplanamadı: "+str(exc))
    try:
        table,reason=observed_information_se(x,items,model)
        if reason:result["messages"].append("SH/GA üretilmedi: "+str(reason))
        if table is not None:
            result["se"]=table
            result["messages"].append(
                "Gözlenen bilgi SH ve Wald GA deneysel kabul edilir; "
                "bağımsız referans doğrulaması bekleniyor.")
    except Exception as exc:
        result["messages"].append("SH/GA üretilmedi: "+str(exc))
    return result
