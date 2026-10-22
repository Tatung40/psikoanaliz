"""B12.6 experimental observed-information audit for mixed IRT.

Finite differences of the analytic negative marginal-score. Standard errors
are withheld for boundary-active solutions, poor curvature, or nonconvergence.
This is NOT yet an independently validated inferential procedure.
"""
import numpy as np
import pandas as pd
from core.irt_mixed import negloglik
from core.irt_mg import Quadrature, _parts, as_matrix

def observed_information(data, groups, fit, *, condition_limit=1e10, step=1e-4):
    if not (np.isfinite(step) and step > 0 and np.isfinite(condition_limit) and condition_limit > 1):
        raise ValueError("Invalid numerical settings.")
    design=fit["design"]
    x,obs=as_matrix(data)
    if len(x)!=fit["n"]:
        raise ValueError("Input and fitted sample sizes disagree.")
    if design.G==1:
        gv=np.zeros(len(x),dtype=int)
    else:
        if groups is None or len(groups)!=len(x):
            raise ValueError("Two-group fits require aligned 0/1 group labels.")
        gv=np.asarray(groups)
        if not np.isin(gv,[0,1]).all() or not np.any(gv==0) or not np.any(gv==1):
            raise ValueError("Groups must contain both 0 and 1.")
    if not fit.get("converged",False):
        return {"valid":False,"reason":"Fit did not meet KKT convergence threshold.","se":None}
    if fit.get("map",False):
        return {"valid":False,"reason":"MAP requires a separately defined posterior-curvature policy.","se":None}
    if fit.get("bound"):
        return {"valid":False,"reason":"Boundary-active estimates: ordinary Wald SE withheld.","se":None}
    v=np.asarray(fit["params"],float)
    if not np.isfinite(v).all() or len(v)!=design.size:
        raise ValueError("Invalid parameter vector.")
    parts=_parts(x,obs,gv,design.G)
    quad=Quadrature(fit["quadrature"],61 if fit["quadrature"]=="mirt" else 41)
    n=len(x)
    h=np.empty((len(v),len(v)),float)
    for j in range(len(v)):
        delta=step*max(1,abs(v[j]))
        plus=v.copy();minus=v.copy()
        plus[j]+=delta;minus[j]-=delta
        gp=negloglik(plus,design,parts,quad,n,with_prior=False)[1]
        gm=negloglik(minus,design,parts,quad,n,with_prior=False)[1]
        h[:,j]=(gp-gm)/(2*delta)*n
    h=(h+h.T)/2
    eig=np.linalg.eigvalsh(h)
    if not np.isfinite(eig).all() or eig[0]<=0:
        return {"valid":False,"reason":"Observed information not positive definite.","se":None,"min_eigenvalue":float(eig[0])}
    cond=float(eig[-1]/eig[0])
    if cond>condition_limit:
        return {"valid":False,"reason":"Observed information ill-conditioned.","se":None,"condition":cond}
    se=np.sqrt(np.diag(np.linalg.inv(h)))
    labels=[f"{p}:{g}:{i}" for p,g,i in design.labels]
    if design.G==2: labels+=["focal_mean","focal_log_sd"]
    return {"valid":True,"reason":"Experimental finite-difference SE; independent validation pending.",
            "condition":cond,"min_eigenvalue":float(eig[0]),
            "se":pd.DataFrame({"parameter":labels,"estimate":v,"se":se})}
