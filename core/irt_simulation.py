"""Reproducible IRT simulation and parameter-recovery diagnostics.

Not a certification of 3PL/4PL calibration accuracy. Recovery must be
assessed across sample sizes, test lengths, true-parameter regimes and seeds.
"""
import numpy as np
import pandas as pd
from core.irt import probabilities, fit_irt
from core.irt_se_diagnostics import hessian_boundary_diagnostics

_MODELS={"1PL":("b",),"2PL":("a","b"),
         "3PL":("a","b","c"),"4PL":("a","b","c","d")}

def simulate_irt(n, true_items, seed=2026):
    """Simulate independent item responses conditional on N(0,1) abilities."""
    if not isinstance(n,(int,np.integer)) or n<30:
        raise ValueError("Örneklem en az 30 kişi olmalıdır.")
    if not isinstance(seed,(int,np.integer)) or seed<0:
        raise ValueError("Tohum negatif olmayan tamsayı olmalıdır.")
    p=pd.DataFrame(true_items)
    if len(p)<3 or p["Madde"].duplicated().any():
        raise ValueError("En az üç farklı madde olmalıdır.")
    a,b,c,d=[p[q].to_numpy(float) for q in ("a","b","c","d")]
    if not (np.isfinite(np.column_stack((a,b,c,d))).all()
            and np.all(a>0) and np.all((c>=0)&(c<d)&(d<=1))):
        raise ValueError("Geçersiz IRT parametreleri.")
    rng=np.random.default_rng(int(seed))
    ability=rng.standard_normal(n)
    pr=probabilities(ability,a,b,c,d)
    responses=pd.DataFrame((rng.random(pr.shape)<pr).astype(int),
                           columns=list(p["Madde"]))
    return responses,ability


def parameter_recovery(data,true_items,model,**fit_kwargs):
    """Return fitted model, recovery table and model-level recovery summary.

    Recovery is descriptive, not proof of convergence or identification.
    Fixed parameters are omitted from active recovery metrics.
    """
    if model not in _MODELS:
        raise ValueError("Bilinmeyen MTK modeli.")
    truth=pd.DataFrame(true_items).reset_index(drop=True)
    if list(data.columns)!=list(truth["Madde"]):
        raise ValueError("Gerçek parametrelerle veri madde sırası uyuşmuyor.")
    result=fit_irt(data,model=model,**fit_kwargs)
    fitted=result["items"].reset_index(drop=True)
    diag=hessian_boundary_diagnostics(
        fitted,model,a_upper=result["a_upper"],d_lower=result["d_lower"])
    records=[]
    for param in _MODELS[model]:
        for j in range(len(truth)):
            value=float(truth.at[j,param])
            estimate=float(fitted.at[j,param])
            records.append({"Madde":truth.at[j,"Madde"],
                            "Parametre":param,"Gerçek":value,
                            "Kestirim":estimate,"Hata":estimate-value,
                            "Mutlak hata":abs(estimate-value)})
    recovery=pd.DataFrame(records)
    summary={
        "model":model,"n":result["n"],"k":result["k"],
        "converged":bool(result["converged"]),
        "iterations":int(result["iterations"]),
        "loglik":float(result["loglik"]),
        "boundary_item_count":int(diag["Madde"].nunique()),
        "boundary_parameter_count":int(len(diag)),
        "MAE":float(recovery["Mutlak hata"].mean()),
        "RMSE":float(np.sqrt(np.mean(recovery["Hata"]**2)))
    }
    for param,part in recovery.groupby("Parametre"):
        summary[f"MAE({param})"]=float(part["Mutlak hata"].mean())
        summary[f"Bias({param})"]=float(part["Hata"].mean())
    return {"fit":result,"recovery":recovery,
            "boundary_diagnostics":diag,"summary":summary}
