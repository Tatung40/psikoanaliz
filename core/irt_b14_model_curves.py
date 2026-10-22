"""B14 fitted category-response curves from Python mirt's model API.

No reconstructed coefficients: use only fitted model probabilities. Refuse
unknown tensor shapes, missing model API or incoherent probabilities.
"""
import numpy as np
import pandas as pd

def fitted_category_curves(result,theta=None):
    fit=result["result"]
    model=getattr(fit,"model",None)
    if model is None or not callable(getattr(model,"probability",None)):
        raise ValueError("Python mirt fitted model probability API unavailable")
    if theta is None:theta=np.linspace(-4,4,161)
    theta=np.asarray(theta,float)
    if theta.ndim!=1 or not np.isfinite(theta).all() or len(theta)<5:
        raise ValueError("Theta grid must be a finite 1-D array")
    # mirt's documented internal probability axes may differ by model.
    raw=np.asarray(model.probability(theta[:,None]),float)
    n,k,c=len(theta),int(result["k"]),len(result["labels"])
    # Interpret only unambiguous axes.
    if raw.shape==(n,k,c):
        p=raw
    elif raw.shape==(k,n,c):
        p=raw.transpose(1,0,2)
    else:
        raise ValueError(f"Unsupported category probability shape {raw.shape}; "
                         f"expected ({n},{k},{c}) or ({k},{n},{c})")
    if not np.isfinite(p).all() or np.min(p)<-1e-9 or np.max(p)>1+1e-9:
        raise ValueError("Invalid fitted probabilities")
    error=float(np.max(np.abs(p.sum(axis=2)-1)))
    if error>1e-6:raise ValueError(f"Category probabilities do not sum to one: {error:g}")
    rows=[]
    for j,name in enumerate(result["items"]):
        for ci,label in enumerate(result["labels"]):
            rows.extend({"Theta":float(th),"Madde":str(name),"Kategori":str(label),
                         "Olasılık":float(pr)} for th,pr in zip(theta,p[:,j,ci]))
    return pd.DataFrame(rows)
