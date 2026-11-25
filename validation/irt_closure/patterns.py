"""Independent pattern integration and exact raw-score inversion; no IRT fitting."""
from functools import lru_cache
import numpy as np
from scipy.special import expit,logsumexp,roots_hermitenorm,log_expit
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq
from scipy.stats import norm
K=12
PATTERNS=((np.arange(1<<K)[:,None]>>np.arange(K))&1).astype(np.uint8)
SCORES=PATTERNS.sum(1).astype(int)
THETA=np.linspace(-4,4,21)


def probability(theta,p):
    return p["c"]+(p["d"]-p["c"])*expit(p["a"]*(np.asarray(theta)[:,None]-p["b"]))


def log_probability(theta,p):
    z=p["a"]*(np.asarray(theta)[:,None]-p["b"])
    with np.errstate(divide="ignore"):
        a=np.logaddexp(np.log(p["c"]),np.log(p["d"]-p["c"])+log_expit(z))
        b=np.logaddexp(np.log1p(-p["d"]),np.log(p["d"]-p["c"])+log_expit(-z))
    return a,b


def moments(p,points=241):
    t,w=roots_hermitenorm(points);w=w/np.sqrt(2*np.pi)
    lp,lq=log_probability(t,p);mass=[];mean=[];sd=[]
    for x in np.array_split(PATTERNS,32):
        ll=x@lp.T+(1-x)@lq.T
        with np.errstate(divide="ignore"): joint=ll+np.log(w)
        z=logsumexp(joint,axis=1);post=np.exp(joint-z[:,None])
        m=post@t;v=post@(t*t)-m*m
        mass.extend(np.exp(z));mean.extend(m);sd.extend(np.sqrt(np.maximum(v,0)))
    return np.array(mass),np.array(mean),np.array(sd)


def cdf_grid(p,step=.01,keep=False):
    t=np.arange(-10,10+step/2,step);lp,lq=log_probability(t,p)
    bounds=[];curves=[] if keep else None
    for x in np.array_split(PATTERNS,32):
        ll=x@np.log(prob).T+(1-x)@np.log1p(-prob).T+norm.logpdf(t)
        pdf=np.exp(ll-ll.max(1,keepdims=True))
        cdf=cumulative_trapezoid(pdf,t,axis=1,initial=0)
        cdf/=cdf[:,-1,None]
        out=[]
        for level in (.025,.975):
            j=np.argmax(cdf>=level,axis=1);j=np.maximum(j,1)
            ii=np.arange(len(x));left=cdf[ii,j-1];right=cdf[ii,j]
            out.append(t[j-1]+(level-left)/(right-left)*(t[j]-t[j-1]))
        bounds.append(np.column_stack(out))
        if keep:curves.append(cdf)
    return t,np.vstack(bounds),np.vstack(curves) if keep else None


def at(cdf,t,v):
    v=np.asarray(v);j=np.searchsorted(t,v,side="right").clip(1,len(t)-1)
    i=np.arange(len(v));a=cdf[i,j-1];b=cdf[i,j]
    y=a+(b-a)*(v-t[j-1])/(t[j]-t[j-1])
    return np.where(v<=t[0],0,np.where(v>=t[-1],1,y))


def score_mass(prob):
    f=np.ones(1)
    for q in prob:f=np.convolve(f,[1-q,q])
    return f


def exact_score_intervals(p):
    minimum=score_mass(p["c"]);maximum=score_mass(p["d"])
    intervals=[]
    for s in range(K+1):
        def upper_tail(t):return score_mass(probability([t],p)[0])[s:].sum()
        def lower_tail(t):return score_mass(probability([t],p)[0])[:s+1].sum()
        if maximum[s:].sum()<.025 or minimum[:s+1].sum()<.025:
            intervals.append((np.inf,-np.inf));continue
        if s==0 or minimum[s:].sum()>=.025:lo=-np.inf
        else:
            left,right=-8.,8.
            while upper_tail(left)>.025:left*=2
            while upper_tail(right)<.025:right*=2
            lo=brentq(lambda t:upper_tail(t)-.025,left,right,xtol=1e-10)
        if s==K or maximum[:s+1].sum()>=.025:hi=np.inf
        else:
            left,right=-8.,8.
            while lower_tail(left)<.025:left*=2
            while lower_tail(right)>.025:right*=2
            hi=brentq(lambda t:lower_tail(t)-.025,left,right,xtol=1e-10)
        intervals.append((lo,hi))
    return np.array(intervals)


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
