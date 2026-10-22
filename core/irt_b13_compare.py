"""B13 guarded cross-engine dichotomous IRT comparison.

Do not compare independently estimated parameter vectors until Python mirt's
reported a,b,c,d reproduce its fitted IRF on a common theta grid.
Rasch is excluded due to distinct free-latent-SD identification.
"""
import numpy as np
import pandas as pd
from scipy.special import expit
from core.irt_python_mirt_adapter import fit_python_mirt

def compare_engines(native, responses):
    model=native["model"]
    if model=="Rasch":raise ValueError("Rasch free-SD latent scale cannot be directly compared to 1PL")
    if model not in ("1PL","2PL","3PL","4PL"):raise ValueError("Unsupported comparison model")
    x=np.asarray(responses,int)
    if x.ndim!=2 or x.shape[1]!=len(native["items"]):raise ValueError("Item count mismatch")
    m=fit_python_mirt(x,model=model).fit
    cs=m.coef_with_se()
    rows=[]
    for name,default in (("discrimination",1.0),("difficulty",0.0),("guessing",0.0),("upper",1.0)):
        if name in cs.columns:
            values=np.asarray(cs[name],float)
        elif name=="difficulty":
            raise ValueError("Python mirt difficulty estimates unavailable")
        else:values=np.full(x.shape[1],default)
        rows.append(values)
    a,b,c,d=rows
    if not all(np.isfinite(q).all() for q in rows) or np.any(a<=0) or np.any(c>=d):
        raise ValueError("Invalid Python mirt parameter estimates")
    theta=np.linspace(-3.5,3.5,29)
    fitted=np.asarray(m.model.probability(theta[:,None]),float).reshape(len(theta),-1)
    formula=c[None,:]+(d-c)[None,:]*expit(a[None,:]*(theta[:,None]-b[None,:]))
    deviation=float(np.max(np.abs(fitted-formula)))
    if fitted.shape!=formula.shape or deviation>1e-8:
        raise ValueError(f"IRF parameter metric is not verified: maximum deviation {deviation:g}")
    pframe=pd.DataFrame({"Madde":native["items"]["Madde"].to_numpy(),
        "a Eser":native["items"]["a"],"a Python mirt":a,
        "b Eser":native["items"]["b"],"b Python mirt":b,
        "c Eser":native["items"]["c"],"c Python mirt":c,
        "d Eser":native["items"]["d"],"d Python mirt":d})
    for key in ("a","b","c","d"):
        pframe[f"Δ{key}"]=pframe[f"{key} Eser"]-pframe[f"{key} Python mirt"]
    summary=[]
    free={"1PL":("b",),"2PL":("a","b"),"3PL":("a","b","c"),"4PL":("a","b","c","d")}[model]
    for key in free:
        delta=pframe[f"Δ{key}"].to_numpy(float)
        summary.append({"Parametre":key,"MAE":float(np.mean(np.abs(delta))),
                        "RMSE":float(np.sqrt(np.mean(delta**2))),
                        "Maks |Δ|":float(np.max(np.abs(delta)))})
    return {"items":pframe,"summary":pd.DataFrame(summary),"irf_deviation":deviation,
            "library_loglik":float(m.log_likelihood),
            "native_loglik":float(native["fit"]["loglik"]),
            "library_converged":bool(m.converged),
            "native_converged":bool(native["fit"]["converged"]),
            "model":model}
