"""Per-item diagnostics for IRT observed-information standard errors.

Identifies directly observable optimization-boundary conditions. Global
information-matrix problems cannot be assigned to one specific item without
further analysis, so those retain a separate model-level warning.
"""
import numpy as np
import pandas as pd

# Shared absolute boundary tolerance for 3PL/4PL diagnostics.
BOUNDARY_TOLERANCE=1e-3

_ACTIVE={"1PL":("b",),"2PL":("a","b"),
         "3PL":("a","b","c"),"4PL":("a","b","c","d")}
_DESCRIPTIONS={
    "a":"Ayırt edicilik (eğimin dikliği)",
    "b":"Güçlük / yetenek konumu",
    "c":"Alt asimptot (düşük yetenekte doğru yanıtlama)",
    "d":"Üst asimptot (yüksek yetenekte doğru yanıtlama)"
}

def hessian_boundary_diagnostics(items, model, a_upper=4., d_lower=.65,
                                  tolerance=BOUNDARY_TOLERANCE):
    """Return explicit at-bound flags using same strict tolerance as Hessian."""
    if model not in _ACTIVE:
        raise ValueError("Bilinmeyen MTK modeli.")
    boundaries={"a":(.15,float(a_upper)), "b":(-4.,4.),
                "c":(0.,.35), "d":(float(d_lower),1.)}
    frame=pd.DataFrame(items)
    rows=[]
    for _,item in frame.iterrows():
        for param in _ACTIVE[model]:
            value=float(item[param])
            low,high=boundaries[param]
            if not np.isfinite(value):
                side="Geçersiz kestirim"
            elif value-low<=tolerance:
                side="Alt sınır"
            elif high-value<=tolerance:
                side="Üst sınır"
            else:
                continue
            rows.append({
                "Madde":str(item["Madde"]),
                "Parametre":param,
                "Parametrenin anlamı":_DESCRIPTIONS[param],
                "Kestirim":value,
                "Alt sınır":low,
                "Üst sınır":high,
                "Sorun":side,
                "SH durumu":"Model düzeyinde hesaplanamaz",
                "Ne yapılmalı":"Kodlama/örneklem ve madde içeriğini kontrol edin; "
                               "daha yalın modelle karşılaştırın; sınırı keyfî değiştirmeyin."
            })
    return pd.DataFrame(rows,columns=[
        "Madde","Parametre","Parametrenin anlamı","Kestirim",
        "Alt sınır","Üst sınır","Sorun","SH durumu","Ne yapılmalı"])
