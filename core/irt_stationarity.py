"""First-order optimality diagnostics for bounded 3PL/4PL MML.

The sign checks implement necessary KKT conditions for maximizing the
marginal log-likelihood, not proof of a global maximum or identification.
The scale is gradient per observation; tolerance is a configurable diagnostic.
"""
import numpy as np
import pandas as pd
from core.irt_hessian import marginal_score
from core.irt import validate_binary
from core.irt_se_diagnostics import BOUNDARY_TOLERANCE

_ACTIVE={"3PL":("a","b","c"),"4PL":("a","b","c","d")}

def stationarity_diagnostics(data, items, model, points=31,
                             a_upper=4.,d_lower=.65,
                             boundary_tolerance=BOUNDARY_TOLERANCE, gradient_tolerance=1e-3):
    if model not in _ACTIVE:
        raise ValueError("Model 3PL/4PL olmalıdır.")
    if gradient_tolerance<=0 or boundary_tolerance<=0:
        raise ValueError("Toleranslar pozitif olmalıdır.")
    x=validate_binary(data)
    table=pd.DataFrame(items).reset_index(drop=True)
    if len(table)!=x.shape[1]:
        raise ValueError("Madde sayıları uyuşmuyor.")
    active=_ACTIVE[model]
    grad=marginal_score(x,table,model,points=points)/len(x)
    bound={"a":(.15,float(a_upper)),"b":(-4.,4.),
           "c":(0.,.35),"d":(float(d_lower),1.)}
    rows=[]
    for j in range(len(table)):
        for m,param in enumerate(active):
            v=float(table.at[j,param])
            derivative=float(grad[j*len(active)+m])
            lo,hi=bound[param]
            if not np.isfinite(derivative):
                violation=float("inf"); position="Geçersiz"
            elif v-lo<=boundary_tolerance:
                position="Alt sınır"
                violation=max(derivative,0.)
            elif hi-v<=boundary_tolerance:
                position="Üst sınır"
                violation=max(-derivative,0.)
            else:
                position="İç bölge"
                violation=abs(derivative)
            rows.append({"Madde":str(table.at[j,"Madde"]),
                         "Parametre":param,"Kestirim":v,
                         "Konum":position,"Marjinal türev/kişi":derivative,
                         "KKT ihlali/kişi":violation,
                         "Tolerans aşıldı":bool(violation>gradient_tolerance)})
    detail=pd.DataFrame(rows)
    summary={
        "model":model,"n":len(x),"k":len(table),
        "max_KKT_violation":float(detail["KKT ihlali/kişi"].max()),
        "violations":int(detail["Tolerans aşıldı"].sum()),
        "parameters":len(detail),
        "gradient_tolerance":float(gradient_tolerance),
        "note":"Tanısal birinci mertebe kontrolüdür; gerçek küresel maksimumu, "
               "model uyumunu veya Hessian SH geçerliğini kanıtlamaz."
    }
    return detail,summary
