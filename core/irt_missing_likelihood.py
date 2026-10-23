"""Missing-aware IRT marginal likelihood, independent of the current complete-case EM.
NaN cells are omitted from each response likelihood (ignorable missingness
conditional on the specified model; not a claim that the data are MCAR).
"""
import numpy as np
from scipy.special import expit, logsumexp
from numpy.polynomial.hermite import hermgauss

def marginal_loglik_missing(data, a, b, c=None, d=None, points=31):
    x=np.asarray(data,dtype=float)
    if x.ndim!=2 or not np.isin(x[np.isfinite(x)],[0,1]).all():
        raise ValueError("Observed responses must be 0/1; missing must be NaN.")
    if np.isinf(x).any():
        raise ValueError("Infinite responses are invalid; use NaN for missing.")
    n,k=x.shape
    a=np.asarray(a,dtype=float)
    b=np.asarray(b,dtype=float)
    c=np.zeros(k) if c is None else np.asarray(c,dtype=float)
    d=np.ones(k) if d is None else np.asarray(d,dtype=float)
    if any(v.shape!=(k,) for v in (a,b,c,d)):
        raise ValueError("Item parameter dimensions differ.")
    gh,w=hermgauss(points)
    theta=np.sqrt(2)*gh
    prior=np.log(w/np.sqrt(np.pi))
    p=np.clip(c[None,:]+(d-c)[None,:]*expit(a[None,:]*(theta[:,None]-b[None,:])),1e-12,1-1e-12)
    observed=np.isfinite(x)
    safe=np.where(observed,x,0.0)
    logresponse=(safe@np.log(p).T + (observed.astype(float)-safe)@np.log1p(-p).T)
    return float(logsumexp(logresponse+prior[None,:],axis=1).sum())
