"""Experimental observed-information uncertainty for dichotomous IRT.

Numerical Hessian of the analytic marginal score, not OPG. No standard
errors are produced if any fitted parameter is at its optimization boundary,
or if the observed information is not demonstrably positive definite.
Requires independent comparison against R mirt before scientific reporting.
"""
import numpy as np
import pandas as pd
from scipy.special import expit,logsumexp
from numpy.polynomial.hermite import hermgauss
from core.irt import validate_binary


def _analytic_marginal_score(x,full,active,theta,logprior):
    """Exact marginal score at supplied quadrature grid and item parameters."""
    a,b,c,d=full.T
    z=expit(theta[:,None]*a[None,:]-a[None,:]*b[None,:])
    p=np.clip(c[None,:]+(d-c)[None,:]*z,1e-10,1-1e-10)
    logpost=x@np.log(p).T+(1-x)@np.log1p(-p).T+logprior[None,:]
    posterior=np.exp(logpost-logsumexp(logpost,axis=1,keepdims=True))
    weighted=posterior.T@x
    total=posterior.sum(axis=0)[:,None]
    scale=(weighted-total*p)/(p*(1-p))
    common=(d-c)[None,:]*z*(1-z)
    deriv={"a":common*(theta[:,None]-b[None,:]),
           "b":-common*a[None,:],"c":1-z,"d":z}
    return np.stack([(scale*deriv[q]).sum(axis=0) for q in active],
                    axis=1).reshape(-1)


def marginal_score(data,items,model,points=31):
    """Analytic marginal log-likelihood score for independent derivative tests."""
    active={"1PL":("b",),"2PL":("a","b"),
            "3PL":("a","b","c"),"4PL":("a","b","c","d")}[model]
    x=validate_binary(data)
    full=np.array([items[q].to_numpy(float) for q in ("a","b","c","d")]).T
    gh,w=hermgauss(points)
    return _analytic_marginal_score(
        x,full,active,np.sqrt(2)*gh,np.log(w/np.sqrt(np.pi)))


def observed_information_se(data,items,model,points=31,a_upper=4.,d_lower=.65,
                            max_parameters=110):
    x=validate_binary(data)
    n,k=x.shape
    active={"1PL":("b",),"2PL":("a","b"),
            "3PL":("a","b","c"),"4PL":("a","b","c","d")}[model]
    m=len(active);count=k*m
    if count>max_parameters:
        return None,"Bilgi matrisi için parametre sayısı işlem sınırını aşıyor."
    bound={"a":(.15,a_upper),"b":(-4.,4.),
           "c":(0.,.35),"d":(d_lower,1.)}
    original=np.array([items[q].to_numpy(float) for q in ("a","b","c","d")]).T
    indices=[("a","b","c","d").index(q) for q in active]
    pars=original[:,indices].reshape(-1)
    for j,q in enumerate(active):
        lo,hi=bound[q]
        vals=pars[j::m]
        if np.any(vals-lo<1e-3) or np.any(hi-vals<1e-3):
            return None,"Parametrelerden biri izin verilen optimizasyon sınırına çok yakın."
    gh,hw=hermgauss(points)
    theta=np.sqrt(2)*gh
    lp=np.log(hw/np.sqrt(np.pi))
    def score(v):
        full=original.copy()
        full[:,indices]=v.reshape(k,m)
        return _analytic_marginal_score(x,full,active,theta,lp)
    hessian=np.zeros((count,count))
    for col in range(count):
        q=active[col%m]
        lo,hi=bound[q]
        step=min(1e-4*max(1,abs(pars[col])),(pars[col]-lo)*.25,
                 (hi-pars[col])*.25)
        if step<=1e-8:
            return None,"Sayısal türev adımı güvenle oluşturulamadı."
        up=pars.copy();down=pars.copy()
        up[col]+=step;down[col]-=step
        hessian[:,col]=-(score(up)-score(down))/(2*step)
    info=(hessian+hessian.T)/2
    eigen=np.linalg.eigvalsh(info)
    if not np.isfinite(eigen).all() or eigen[0]<=1e-7:
        return None,"Gözlenen bilgi matrisi pozitif tanımlı değil; SH üretilemez."
    cond=float(eigen[-1]/eigen[0])
    if cond>1e8:
        return None,f"Gözlenen bilgi matrisi kötü koşullanmış (koşul sayısı {cond:.3g})."
    try:
        cov=np.linalg.inv(info)
    except np.linalg.LinAlgError:
        return None,"Gözlenen bilgi matrisi terslenemedi."
    variances=np.diag(cov)
    if np.any(variances<=0) or not np.isfinite(variances).all():
        return None,"Pozitif, sonlu SH varyansı hesaplanamadı."
    se=np.sqrt(variances).reshape(k,m)
    out=pd.DataFrame({"Madde":list(items["Madde"])})
    for col,q in enumerate(active):
        out["SH("+q+")"]=se[:,col]
        out[q+" %95 alt"]=items[q].to_numpy(float)-1.96*se[:,col]
        out[q+" %95 üst"]=items[q].to_numpy(float)+1.96*se[:,col]
    return out,""
