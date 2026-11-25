"""Validation aliases preserve the independently checked production kernels."""
from functools import lru_cache
import numpy as np
from scipy.stats import norm
from core.irt_binary_patterns import K,PATTERNS,SCORES,THETA
from core.irt_binary_patterns import _probability as probability, _log_probability as log_probability, _moments as moments, _cdf_grid as cdf_grid, _at as at, _score_mass as score_mass, _exact_score_intervals as exact_score_intervals
@lru_cache(2)
def oracle(model):
    from validation.binary_certification.common import truth
    p=truth(model,K);mass,mean,sd=moments(p,481)
    t,bounds,cdf=cdf_grid(p,.005,True)
    lp,lq=log_probability(THETA,p)
    conditional=np.exp(PATTERNS@lp.T+(1-PATTERNS)@lq.T)
    score_ci=exact_score_intervals(p)[SCORES]
    return p,mass,mean,sd,t,bounds,cdf,conditional,score_ci


def interval_metrics(bounds,reference):
    _,mass,_,_,t,_,cdf,conditional,_=reference
    lo,hi=bounds.T
    population=float(mass@np.where(lo<=hi,at(cdf,t,hi)-at(cdf,t,lo),0))
    grid=np.sum(conditional*((lo[:,None]<=THETA)&(THETA<=hi[:,None])),axis=0)
    tails=[]
    for left,right in ((-np.inf,-2.),(2.,np.inf)):
        a=np.maximum(lo,left);b=np.minimum(hi,right)
        value=np.where(a<=b,at(cdf,t,b)-at(cdf,t,a),0)
        tails.append(float(mass@value/(norm.cdf(right)-norm.cdf(left))))
    return {"marginal":population,"conditional_theta":grid.tolist(),"tails":tails}


def person_evidence(model,p,native_mean,native_sd):
    ref=oracle(model);_,mass,meanK,sdK,_,boundsK,_,_,scoreK=ref
    _,meanF,sdF=moments(p,241);_,meanFine,sdFine=moments(p,481)
    delta=max(float(abs(meanF-native_mean).max()),float(abs(sdF-native_sd).max()),
              float(abs(meanF-meanFine).max()),float(abs(sdF-sdFine).max()))
    if delta>1e-6:raise ValueError("Person posterior moment audit")
    _,coarse,_=cdf_grid(p,.01)
    _,bounds,_=cdf_grid(p,.005)
    if abs(bounds-coarse).max()>1e-3:raise ValueError("Posterior quantile grid audit")
    coarse_coverage=interval_metrics(coarse,ref)["marginal"]
    credible=interval_metrics(bounds,ref)
    if abs(credible["marginal"]-coarse_coverage)>1e-4:raise ValueError("Posterior coverage grid audit")
    oracle_mse=float(mass@(sdK*sdK))
    mse=float(mass@((meanF-meanK)**2+sdK**2))
    scoreF=exact_score_intervals(p)[SCORES]
    bias=float(mass@(meanF-meanK))
    var=float(mass@(meanF-float(mass@meanF))**2)
    return {"bias":bias,"rmse":float(np.sqrt(mse)),"mse":mse,"oracle_mse":oracle_mse,
        "excess_mse_fraction":mse/oracle_mse-1,
        "correlation":float(mass@(meanF*meanK)/np.sqrt(var)),
        "credible_F":credible,"credible_K":interval_metrics(boundsK,ref),
        "exact_score_F":interval_metrics(scoreF,ref),"exact_score_K":interval_metrics(scoreK,ref),
        "max_moment_audit":delta,"max_quantile_audit":float(abs(bounds-coarse).max())}
