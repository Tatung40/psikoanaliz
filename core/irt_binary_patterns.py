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


def _probability(theta,p):
    return p["c"]+(p["d"]-p["c"])*expit(p["a"]*(np.asarray(theta)[:,None]-p["b"]))


def _log_probability(theta,p):
    z=p["a"]*(np.asarray(theta)[:,None]-p["b"])
    with np.errstate(divide="ignore"):
        a=np.logaddexp(np.log(p["c"]),np.log(p["d"]-p["c"])+log_expit(z))
        b=np.logaddexp(np.log1p(-p["d"]),np.log(p["d"]-p["c"])+log_expit(-z))
    return a,b


def _moments(p,points=241):
    t,w=roots_hermitenorm(points);w=w/np.sqrt(2*np.pi)
    lp,lq=_log_probability(t,p);mass=[];mean=[];sd=[]
    for x in np.array_split(PATTERNS,32):
        ll=x@lp.T+(1-x)@lq.T
        with np.errstate(divide="ignore"): joint=ll+np.log(w)
        z=logsumexp(joint,axis=1);post=np.exp(joint-z[:,None])
        m=post@t;v=post@(t*t)-m*m
        mass.extend(np.exp(z));mean.extend(m);sd.extend(np.sqrt(np.maximum(v,0)))
    return np.array(mass),np.array(mean),np.array(sd)


def _cdf_grid(p,step=.01,keep=False):
    t=np.arange(-10,10+step/2,step);lp,lq=_log_probability(t,p)
    bounds=[];curves=[] if keep else None
    for x in np.array_split(PATTERNS,32):
        ll=x@lp.T+(1-x)@lq.T+norm.logpdf(t)
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


def _at(cdf,t,v):
    v=np.asarray(v);j=np.searchsorted(t,v,side="right").clip(1,len(t)-1)
    i=np.arange(len(v));a=cdf[i,j-1];b=cdf[i,j]
    y=a+(b-a)*(v-t[j-1])/(t[j]-t[j-1])
    return np.where(v<=t[0],0,np.where(v>=t[-1],1,y))


def _score_mass(prob):
    f=np.ones(1)
    for q in prob:f=np.convolve(f,[1-q,q])
    return f


def _exact_score_intervals(p):
    minimum=_score_mass(p["c"]);maximum=_score_mass(p["d"])
    intervals=[]
    for s in range(K+1):
        def upper_tail(t):return _score_mass(_probability([t],p)[0])[s:].sum()
        def lower_tail(t):return _score_mass(_probability([t],p)[0])[:s+1].sum()
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



# Exact same numerical kernels as the independently confirmed validation study.
_CERT_PATH=__import__("pathlib").Path(__file__).resolve().parents[1]/"validation/evidence/irt_closure/P2_CERTIFICATION.json"
METHOD_VERSION="pattern-person-v1"


@lru_cache(1)
def _certificate():
    import json
    try:
        value=json.loads(_CERT_PATH.read_text())
        return value if value.get("method_version")==METHOD_VERSION else {}
    except (OSError,ValueError):return {}


def _eligible(outcome):
    if outcome.model not in ("3PL","4PL") or getattr(outcome,"estimation",None)!="MAP":
        return False
    settings=getattr(outcome,"solver_settings",{})
    if (outcome.n_persons,outcome.n_items)!=(3000,12) or settings!=dict(max_iter=500,tol=1e-4,n_quadpts=61):
        return False
    import importlib.metadata
    if importlib.metadata.version("mirt")!="1.2.0":return False
    model=_certificate().get("models",{}).get(outcome.model,{})
    return bool(model.get("recovery") and model.get("credible_marginal") and model.get("score_interval"))


@lru_cache(4)
def _interval_lookup(model,key):
    values=np.array(key,float).reshape(4,12)
    p={q:values[i] for i,q in enumerate("abcd")}
    if not all(np.isfinite(v).all() for v in p.values()) or np.any(p["a"]<=0) or np.any(p["c"]<0) or np.any(p["d"]>1) or np.any(p["c"]>=p["d"]):
        raise ValueError("Geçerli pozitif ayırıcılık ve c<d olasılıkları gerekir.")
    _,coarse,_=_cdf_grid(p,.01)
    _,credible,_=_cdf_grid(p,.005)
    if np.max(abs(credible-coarse))>1e-3:
        raise ValueError("Posterior aralık ızgara denetimi geçmedi.")
    score=_exact_score_intervals(p)
    credible.setflags(write=False);score.setflags(write=False)
    return credible,score


def _append_person_intervals(frame,outcome,responses):
    """Scope-gated intervals; native GH241 point/SD and IDs remain unchanged."""
    if not _eligible(outcome):return frame
    co=outcome.fit.coef()
    p=[co[name].to_numpy(float) if name in co else np.ones(12)
       for name in ("discrimination","difficulty","guessing","upper")]
    try:credible,score=_interval_lookup(outcome.model,tuple(np.concatenate(p)))
    except (ValueError,ArithmeticError) as exc:
        result=frame.copy();result["Aralık notu"]="Aralıklar üretilmedi: "+str(exc)
        return result
    x=np.asarray(responses,int)
    code=(x*(1<<np.arange(12))).sum(1)
    raw=x.sum(1);selected=score[raw];empty=selected[:,0]>selected[:,1]
    result=frame.copy()
    result["Posterior %95 alt"]=credible[code,0]
    result["Posterior %95 üst"]=credible[code,1]
    result["Skor GA %95 alt"]=np.where(empty,np.nan,selected[:,0])
    result["Skor GA %95 üst"]=np.where(empty,np.nan,selected[:,1])
    result["Güven kümesi boş"]=empty
    result["Posterior aralık türü"]="Eşit kuyruklu Bayesçi; sabit θ için sıklıkçı kapsama garantisi değildir"
    result["Skor aralık türü"]="Tam Poisson-binomial tersleme; madde kestirimlerine koşullu"
    result["Bilimsel durum"]="SCOPED CERTIFIED — referans N3000/J12, MAP/GH61, N(0,1)"
    result["Doğrulama kapsamı"]="Önkayıtlı madde düzeni; θ -4..4 (21 nokta) ve N(0,1) kuyruk ortalamaları; J30 doğrulanmadı"
    return result


def independent_person_scope(outcome):
    """Reference-scoped eligibility, never a proof of population assumptions."""
    return _eligible(outcome)


def score_independent_persons(outcome,responses,identifiers,calibration_identifiers,independence_declared=False):
    """Validated held-out scoring only; never infer independence from shape/hash."""
    if not independence_declared:
        raise ValueError("Kalibrasyondan bağımsız birey örneklemi açıkça beyan edilmelidir.")
    if not _eligible(outcome):
        raise ValueError("Bağımsız aralık doğrulaması yalnız MAP/GH61 N3000/J12 referans kapsamındadır.")
    import pandas as pd
    if identifiers is None or calibration_identifiers is None or any(pd.isna(v) for v in list(identifiers)+list(calibration_identifiers)):
        raise ValueError("Eksik birey kimlikleriyle bağımsız çıkarım yapılamaz.")
    training=[str(v).strip() for v in calibration_identifiers]
    ids=[str(v).strip() for v in identifiers]
    if len(training)!=outcome.n_persons or len(set(training))!=len(training) or any(not v or v.lower()=="nan" for v in training):
        raise ValueError("Kalibrasyon kimlikleri eksiksiz ve benzersiz olmalıdır.")
    if len(ids)!=len(responses) or len(set(ids))!=len(ids) or any(not v or v.lower()=="nan" for v in ids):
        raise ValueError("Bağımsız birey kimlikleri eksiksiz ve benzersiz olmalıdır.")
    if set(training)&set(ids):
        raise ValueError("Puanlama ve kalibrasyon kimlikleri çakışıyor; bağımsız çıkarım kapalı.")
    from core.irt_binary_persons import binary_person_abilities
    frame=binary_person_abilities(outcome,responses,identifiers=ids)
    result=_append_person_intervals(frame,outcome,responses)
    if "Posterior %95 alt" not in result:
        raise ValueError("Aralık sayısal denetimi geçmedi; bağımsız çıkarım üretilmedi.")
    result["Bağımsızlık koşulu"]="Kullanıcı beyanı ve kalibrasyonla ayrık birey kimlikleri"
    return result
