"""Missing-aware MML/EM calibration for dichotomous 1PL–4PL.
Experimental: do not use for inferential publication until three-way validation.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from numpy.polynomial.hermite import hermgauss
from core.irt import probabilities, _initial, _unpack, _bounds, structural_identifiability
from core.irt_missing_likelihood import marginal_loglik_missing

def fit_irt_missing(data, model="2PL", points=21, max_iter=60, tolerance=1e-5):
    if model not in ("1PL","2PL","3PL","4PL"):
        raise ValueError("Unknown model")
    x=np.asarray(data,dtype=float)
    if x.ndim!=2 or x.shape[0]<30 or x.shape[1]<3:
        raise ValueError("Need at least 30 people and 3 items")
    observed=np.isfinite(x)
    if not np.isin(x[observed],[0,1]).all():
        raise ValueError("Observed responses must be 0/1")
    n,k=x.shape
    if not structural_identifiability(k,model)["satisfied"]:
        raise ValueError("Model not structurally identifiable")
    counts=observed.sum(0)
    if np.any(counts<10):
        raise ValueError("At least 10 observed responses required per item")
    pop=np.nansum(x,axis=0)/counts
    if np.any((pop<=.01)|(pop>=.99)):
        raise ValueError("Item observed proportions must lie inside (.01,.99)")
    params=[_initial(float(p),model) for p in pop]
    bounds=[(-4.,4.)] if model=="1PL" else _bounds(model)
    nodes,weights=hermgauss(points)
    nodes=np.sqrt(2)*nodes
    logprior=np.log(weights/np.sqrt(np.pi))
    safe=np.where(observed,x,0.)
    obsfloat=observed.astype(float)
    history=[]
    converged=False
    for step in range(max_iter):
        a,b,c,d=(np.array(v) for v in zip(*[_unpack(t,model) for t in params]))
        p=np.clip(probabilities(nodes,a,b,c,d),1e-10,1-1e-10)
        lp=safe@np.log(p).T+(obsfloat-safe)@np.log1p(-p).T+logprior[None,:]
        norm=logsumexp(lp,axis=1)
        posterior=np.exp(lp-norm[:,None])
        ll=float(norm.sum())
        successes=safe.T@posterior
        trials=obsfloat.T@posterior
        nextparams=[]
        for j in range(k):
            s=successes[j]
            t=trials[j]
            def objective(par):
                aj,bj,cj,dj=_unpack(par,model)
                pj=np.clip(probabilities(nodes,np.array([aj]),np.array([bj]),
                        np.array([cj]),np.array([dj]))[:,0],1e-10,1-1e-10)
                return -float(s@np.log(pj)+(t-s)@np.log1p(-pj))
            res=minimize(objective,params[j],bounds=bounds,method="L-BFGS-B",
                         options={"maxiter":100,"ftol":1e-10})
            nextparams.append(res.x if np.isfinite(res.fun) else params[j])
        params=nextparams
        a,b,c,d=(np.array(v) for v in zip(*[_unpack(t,model) for t in params]))
        newll=marginal_loglik_missing(x,a,b,c,d,points=points)
        history.append(newll)
        if newll<ll-1e-5*(1+abs(ll)):
            raise RuntimeError("EM likelihood decreased materially")
        if abs(newll-ll)<tolerance:
            converged=True
            break
    table=pd.DataFrame({"a":a,"b":b,"c":c,"d":d})
    return {"model":model,"items":table,"loglik":newll,
            "aic":2*k*len(params[0])-2*newll,
            "bic":np.log(n)*k*len(params[0])-2*newll,
            "converged":converged,"iterations":len(history),
            "history":history,"missing_count":int((~observed).sum()),
            "observed_count":int(observed.sum()),"n":n,"k":k,
            "status":"experimental_requires_three_way_validation"}
