"""Quadrature sensitivity for bounded 3PL/4PL marginal ML fits."""
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp
from core.irt import probabilities,validate_binary
from core.irt_multistart import fit_irt_multistart

def marginal_loglik_at_fixed_parameters(data,items,points=61):
    """Independent fixed-parameter marginal log-likelihood evaluation."""
    if not 9<=points<=121:
        raise ValueError("Quadrature point count out of range.")
    x=validate_binary(data)
    table=pd.DataFrame(items)
    if len(table)!=x.shape[1]:
        raise ValueError("Item counts differ.")
    a,b,c,d=[table[p].to_numpy(float) for p in ("a","b","c","d")]
    gh,w=hermgauss(points)
    p=np.clip(probabilities(np.sqrt(2)*gh,a,b,c,d),1e-12,1-1e-12)
    ll=logsumexp(x@np.log(p).T+(1-x)@np.log1p(-p).T
                 +np.log(w/np.sqrt(np.pi))[None,:],axis=1).sum()
    return float(ll)

def quadrature_sensitivity(data,model,fit_points=(15,21,31,41),
                           reference_points=61,max_iter=70,starts=(0.,)):
    if model not in ("3PL","4PL"):
        raise ValueError("Only 3PL and 4PL supported.")
    if not fit_points or max(fit_points)>=reference_points:
        raise ValueError("Reference grid must be finer than fit grids.")
    runs=[]; parameter_tables={}
    for pts in fit_points:
        audit=fit_irt_multistart(
            data,model=model,starts=starts,points=int(pts),
            max_iter=max_iter)
        fit=audit["fit"]
        ll=marginal_loglik_at_fixed_parameters(
            data,fit["items"],points=reference_points)
        runs.append({"Model":model,"Fit düğümü":pts,
                     "Karşılaştırma düğümü":reference_points,
                     "EM yakınsaması":fit["converged"],
                     "Fit logLik":fit["loglik"],
                     "Ortak ağ logLik":ll,
                     "Sınırdaki parametre":len(audit["boundary"])})
        parameter_tables[pts]=fit["items"][["a","b","c","d"]].to_numpy(float)
    baseline=parameter_tables[max(fit_points)]
    for row in runs:
        p=parameter_tables[row["Fit düğümü"]]
        row["Ortak ağ referansa parametre farkı ort."]=float(
            np.mean(np.abs(p-baseline)))
    return pd.DataFrame(runs)
