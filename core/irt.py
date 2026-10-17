"""Tek boyutlu ikili MTK: Gauss-Hermite MML/EM kalibrasyonu (1PL-4PL)."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from numpy.polynomial.hermite import hermgauss
from core.irt_se_diagnostics import hessian_boundary_diagnostics, BOUNDARY_TOLERANCE

def probabilities(theta, a, b, c, d):
    """P(X=1|theta)=c+(d-c)*sigmoid(a*(theta-b)); theta x item."""
    z = expit(np.asarray(theta, dtype=float)[:, None] * a[None, :] - (a*b)[None, :])
    return c[None, :] + (d-c)[None, :] * z

def information(theta, a, b, c, d):
    z = expit(np.asarray(theta, dtype=float)[:, None]*a[None, :]-(a*b)[None, :])
    p = np.clip(c[None, :] + (d-c)[None, :]*z, 1e-10, 1-1e-10)
    derivative = (d-c)[None, :]*a[None, :]*z*(1-z)
    return derivative**2/(p*(1-p))

def validate_binary(data):
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[0] < 30 or x.shape[1] < 3:
        raise ValueError("MTK için en az 30 katılımcı ve 3 madde gereklidir.")
    if not np.isfinite(x).all() or not np.isin(x, [0, 1]).all():
        raise ValueError("MTK, eksiksiz 0/1 puanlı maddeler gerektirir. Eksikleri önce temizleyin.")
    p = x.mean(axis=0)
    if np.any((p <= .01) | (p >= .99)):
        raise ValueError("Ortalaması 0.01–0.99 dışında olan maddeleri ayıklayın (tüm doğru/tüm yanlış dahil).")
    return x

def _bounds(model):
    b = [(0.15, 4.0), (-4.0, 4.0)]
    if model in ("3PL", "4PL"):
        b.append((0.0, .35))
    if model == "4PL":
        b.append((.65, 1.0))
    return b

def _unpack(p, model):
    a,b = (1.0, p[0]) if model=="1PL" else (p[0],p[1])
    c = p[2] if model in ("3PL", "4PL") else 0.
    d = p[3] if model=="4PL" else 1.
    return float(a), float(b), float(c), float(d)

def _initial(prob, model):
    b = float(np.clip(-np.log(prob/(1-prob)), -3, 3))
    if model=="1PL": return np.array([b])
    if model=="2PL": return np.array([1., b])
    if model=="3PL": return np.array([1., b, .1])
    return np.array([1., b, .1, .95])

_ACTIVE = {"1PL": ("b",), "2PL": ("a", "b"), "3PL": ("a", "b", "c"),
           "4PL": ("a", "b", "c", "d")}


def structural_identifiability(k, model):
    """Necessary (not sufficient) parameter-counting condition.

    Complete 0/1 data on k items have 2**k response patterns, i.e. at most
    2**k - 1 free pattern probabilities. A model with more free item
    parameters cannot be identified whatever the data, starting values or
    item priors (a prior regularizes the estimate; it adds no information
    about the response patterns). Equality is not rejected automatically.
    Satisfying the condition does not prove identification.
    """
    if model not in _ACTIVE:
        raise ValueError("Geçersiz MTK modeli.")
    k = int(k)
    parameters = k * len(_ACTIVE[model])
    pattern_df = 2**k - 1
    reason = "" if parameters <= pattern_df else (
        f"{model} modeli {k} maddede {parameters} serbest parametre içerir; tam 0/1 "
        f"veride en fazla {pattern_df} bağımsız yanıt örüntüsü olasılığı vardır. Model "
        "yapısal olarak tanımlanamaz; madde sayısını artırın veya daha yalın bir model seçin.")
    return {"parameters": parameters, "pattern_df": pattern_df,
            "satisfied": not reason, "reason": reason}


def fit_irt(data, model="2PL", points=31, max_iter=70, tolerance=1e-4,
            a_upper=4.0, d_lower=0.65, initial_shift=0.0, param_tolerance=1e-3):
    """MML EM, N(0,1) özdeşleme; 1PL için a=1 (Rasch benzeri sabit eğim).
    Returns item params, EAP theta, posterior SD, diagnostic/log-likelihood.

    `converged` keeps the marginal log-likelihood criterion (|dlogL| <
    tolerance). `convergence` additionally reports the largest absolute
    parameter change in the last EM cycle against `param_tolerance`
    (diagnostic only; it does not change `converged`), separately for
    parameters inside the bounds and parameters on an optimization bound.
    Models with more free parameters than 2**k - 1 are rejected.
    """
    if model not in ("1PL","2PL","3PL","4PL"):
        raise ValueError("Geçersiz MTK modeli.")
    x = validate_binary(data)
    n,k = x.shape
    structure = structural_identifiability(k, model)
    if not structure["satisfied"]:
        raise ValueError(structure["reason"])
    if not (np.isfinite(param_tolerance) and param_tolerance > 0):
        raise ValueError("Parametre değişimi toleransı pozitif olmalıdır.")
    gh, hw = hermgauss(points)
    nodes = np.sqrt(2)*gh
    logprior = np.log(hw/np.sqrt(np.pi))
    pop = x.mean(0)
    if not (4.0 <= a_upper <= 30.0 and 0.05 <= d_lower <= 0.65):
        raise ValueError("Parametre sınırları geçersiz.")
    params = [_initial(p,model) for p in pop]
    bounds = [(-4,4)] if model=="1PL" else _bounds(model)
    if model != "1PL":
        bounds[0] = (0.15, a_upper)
    if model == "4PL":
        bounds[-1] = (d_lower, 1.0)
    if initial_shift and model in ("3PL","4PL"):
        for par in params:
            par[0] = float(np.clip(1+initial_shift*.4, 0.16, a_upper-.01))
            par[2] = float(np.clip(.1+initial_shift*.08, .001, .34))
            if model=="4PL":
                par[3] = float(np.clip(.95-initial_shift*.15, d_lower+.01, .99))
    history=[]
    param_changes=[]
    last_delta=None
    last_ll_change=np.nan
    converged=False
    for step in range(max_iter):
        a,b,c,d = (np.array(v) for v in zip(*[_unpack(t,model) for t in params]))
        probs = np.clip(probabilities(nodes,a,b,c,d), 1e-9, 1-1e-9)
        logpost = x@np.log(probs).T + (1-x)@np.log1p(-probs).T + logprior[None,:]
        lognorm = logsumexp(logpost,axis=1)
        ll = float(lognorm.sum())
        history.append(ll)
        posterior = np.exp(logpost-lognorm[:,None])
        weight = posterior.sum(0)
        successes = x.T @ posterior
        nextparams=[]
        for j in range(k):
            s = successes[j]
            def objective(par):
                aj,bj,cj,dj = _unpack(par,model)
                pj = np.clip(probabilities(nodes, np.array([aj]),np.array([bj]),
                      np.array([cj]),np.array([dj]))[:,0],1e-9,1-1e-9)
                return -float(s@np.log(pj)+(weight-s)@np.log1p(-pj))
            result=minimize(objective,params[j],method="L-BFGS-B",bounds=bounds,
                            options={"maxiter":85,"ftol":1e-9})
            nextparams.append(result.x if np.isfinite(result.fun) else params[j])
        last_delta=np.abs(np.vstack(nextparams)-np.vstack(params))
        param_changes.append(float(last_delta.max()))
        params=nextparams
        # Yakınsama kararını yeni M-adımı parametrelerinde hesaplanan
        # marjinal log-olabilirliğe göre ver; önceki E-adımlarını kıyaslama.
        a1,b1,c1,d1=(np.array(v) for v in zip(*[_unpack(t,model) for t in params]))
        updated_probs=np.clip(probabilities(nodes,a1,b1,c1,d1),1e-9,1-1e-9)
        updated_logpost=(x@np.log(updated_probs).T
                         +(1-x)@np.log1p(-updated_probs).T+logprior[None,:])
        updated_ll=float(logsumexp(updated_logpost,axis=1).sum())
        if updated_ll < ll - 1e-5 * (1+abs(ll)):
            # M-adımı beklenen fonksiyonu artırdığı için büyük düşüş olası
            # sayısal hata veya yetersiz optimizasyona işaret edebilir.
            raise RuntimeError("EM log-olabilirliği azaldı; sayısal yakınsama güvenilir değil.")
        last_ll_change=abs(updated_ll-ll)
        if abs(updated_ll-ll) < tolerance:
            converged=True
            history.append(updated_ll)
            break
    a,b,c,d=(np.array(v) for v in zip(*[_unpack(t,model) for t in params]))
    probs=np.clip(probabilities(nodes,a,b,c,d),1e-9,1-1e-9)
    logpost=x@np.log(probs).T+(1-x)@np.log1p(-probs).T+logprior[None,:]
    lnorm=logsumexp(logpost,axis=1)
    post=np.exp(logpost-lnorm[:,None])
    eap=post@nodes
    psd=np.sqrt(np.maximum(post@(nodes**2)-eap**2,0))
    ll=float(lnorm.sum())
    numpar=k*({"1PL":1,"2PL":2,"3PL":3,"4PL":4}[model])
    theta=np.linspace(-4,4,161)
    iteminfo=information(theta,a,b,c,d)
    # Posterior predictive model fit: empirical and expected proportion; descriptive only
    expected=post@probs
    residual=x-expected
    table=pd.DataFrame({"Madde":list(data.columns) if isinstance(data,pd.DataFrame) else
                       [f"M{i+1:02d}" for i in range(k)],
                       "a":a,"b":b,"c":c,"d":d,"Gözlenen p":pop,
                       "Model p":expected.mean(0),
                       "Artık RMS":np.sqrt(np.mean(residual**2,axis=0)),
                       "Sınır uyarısı":[bool((model!="1PL" and (a[i]<.17 or a[i]>a_upper-.05)) or
                           abs(b[i])>3.95 or (model in ("3PL","4PL") and c[i]>.345) or
                           (model=="4PL" and d[i]<d_lower+.005)) for i in range(k)]})
    # Strict boundary diagnostics (same rule as the Hessian SE gate), by position.
    diag=hessian_boundary_diagnostics(table.assign(Madde=np.arange(k)),model,
                                      a_upper=a_upper,d_lower=d_lower)
    active=_ACTIVE[model]
    at_bound=np.zeros((k,len(active)),dtype=bool)
    for i,param in zip(diag["Madde"].astype(int),diag["Parametre"]):
        at_bound[i,active.index(param)]=True
    if model in ("3PL","4PL"):
        table["Sınır uyarısı"]=at_bound.any(axis=1)
    convergence=_convergence_diagnostics(converged,max_iter,tolerance,last_ll_change,
                                         param_tolerance,last_delta,at_bound,
                                         table["Madde"].tolist(),active)
    warnings=[]
    if convergence["status"]!="converged" and converged:
        warnings.append(convergence["message"])
    if at_bound.any():
        warnings.append(f"{int(at_bound.sum())} parametre optimizasyon sınırında "
                        f"({int(at_bound.any(axis=1).sum())}/{k} madde). Bu parametreler sınır "
                        "kestirimidir: Hessian SH ve Wald aralıkları hesaplanmaz; sınırı keyfî "
                        "genişletmek tanımlanabilirliği sağlamaz.")
    return {"model":model,"items":table,"theta":eap,"theta_sd":psd,
            "grid":theta,"icc":probabilities(theta,a,b,c,d),
            "iif":iteminfo,"tif":iteminfo.sum(axis=1),
            "loglik":ll,"aic":2*numpar-2*ll,
            "bic":np.log(n)*numpar-2*ll,
            "iterations":len(history),"converged":converged,
            "history":history,"n":n,"k":k,"param_count":numpar,
            "a_upper":a_upper,"d_lower":d_lower,"initial_shift":initial_shift,
            "param_change_history":param_changes,"convergence":convergence,
            "structure":structure,"warnings":warnings,
            "warning":("4PL modelindeki c ve d parametreleri yüksek örneklem büyüklüğü "
                       "ve güçlü madde bilgisi gerektirir; sınır kestirimlerini inceleyin."
                       if model=="4PL" else "")}


def _convergence_diagnostics(converged, max_iter, tolerance, ll_change,
                             param_tolerance, delta, at_bound, names, active):
    """Log-likelihood criterion plus last-cycle parameter change (diagnostic).

    A parameter fixed on an optimization bound has zero change once pinned; a
    large change of a parameter that has just reached a bound is reported
    separately, since it signals a boundary estimate rather than slow EM.
    """
    if delta is None:
        delta=np.zeros_like(at_bound,dtype=float)
    interior=delta[~at_bound]
    bounded=delta[at_bound]
    max_change=float(delta.max()) if delta.size else 0.
    worst=np.unravel_index(int(np.argmax(delta)),delta.shape) if delta.size else (0,0)
    params_ok=bool(max_change<=param_tolerance)
    if not converged:
        status="not_converged"
        message=(f"logL ölçütü (|ΔlogL| < {tolerance:g}) {max_iter} yinelemede sağlanmadı; "
                 "kestirimler geçicidir.")
    elif params_ok:
        status="converged"
        message="logL ve parametre değişimi ölçütleri sağlandı."
    else:
        status="loglik_only"
        where=f"{names[worst[0]]} maddesinin {active[worst[1]]} parametresi"
        if at_bound[worst]:
            message=(f"logL ölçütü sağlandı, ancak son EM yinelemesinde {where} "
                     f"{max_change:.2g} değişerek optimizasyon sınırına ulaştı "
                     f"(eşik {param_tolerance:g}). Bu bir sınır kestirimidir; KKT tanısı "
                     "(stationarity_diagnostics) ve daha yalın modelle karşılaştırma önerilir.")
        else:
            message=(f"logL ölçütü sağlandı, ancak son EM yinelemesinde {where} hâlâ "
                     f"{max_change:.2g} değişti (eşik {param_tolerance:g}). Olabilirlik yüzeyi "
                     "düz olabilir (zayıf tanımlanabilirlik) veya EM yavaş ilerliyor; daha küçük "
                     "tolerans ve daha fazla yinelemeyle sonuçların değişmediğini doğrulayın.")
    return {"status":status,"message":message,"loglik_converged":bool(converged),
            "loglik_tolerance":float(tolerance),"last_loglik_change":float(ll_change),
            "param_tolerance":float(param_tolerance),"max_param_change":max_change,
            "max_param_change_interior":float(interior.max()) if interior.size else 0.,
            "max_param_change_at_bound":float(bounded.max()) if bounded.size else 0.,
            "parameters_at_bound":int(at_bound.sum()),
            "parameter_criterion_met":params_ok}


def estimate_ability(data, item_table, method="EAP", points=81, prior_sd=1.0):
    """Estimate person ability under fixed item params. EAP/MAP use N(0, prior_sd²);
    ML uses likelihood only and flags boundary estimates.
    Scores are conditional on fitted item parameters; SE is posterior SD (EAP),
    local curvature approximation (MAP/ML).
    """
    from scipy.special import logsumexp
    x = validate_binary(data)
    k = x.shape[1]
    if len(item_table) != k:
        raise ValueError("Madde sayısı parametre tablosuyla uyuşmuyor.")
    if method not in ("EAP","MAP","ML"):
        raise ValueError("Yetenek kestirim yöntemi EAP, MAP veya ML olmalıdır.")
    if method != "ML" and (prior_sd <= 0 or not np.isfinite(prior_sd)):
        raise ValueError("Önsel standart sapma pozitif olmalıdır.")
    a,b,c,d = [np.asarray(item_table[col], dtype=float) for col in ("a","b","c","d")]
    theta = np.linspace(-5.,5.,int(points))
    p = np.clip(probabilities(theta,a,b,c,d),1e-10,1-1e-10)
    loglike = x@np.log(p).T+(1-x)@np.log1p(-p).T
    logprior = -.5*(theta/prior_sd)**2 if method in ("EAP","MAP") else np.zeros_like(theta)
    lp = loglike + logprior[None,:]
    if method=="EAP":
        weights = np.exp(lp-logsumexp(lp,axis=1,keepdims=True))
        ability=weights@theta
        se=np.sqrt(np.maximum(weights@(theta**2)-ability**2,0))
        bounds=np.zeros(len(x),dtype=bool)
    else:
        index=np.argmax(lp,axis=1)
        ability=theta[index]
        bounds=(index==0)|(index==len(theta)-1)
        # Approximate conditional SE from Fisher information and prior precision.
        iif=information(ability,a,b,c,d).sum(axis=1)
        prior_precision=(1/prior_sd**2) if method=="MAP" else 0.
        se=1/np.sqrt(np.maximum(iif+prior_precision,1e-10))
    return pd.DataFrame({"Yetenek (theta)":ability,
                         "Standart Hata":se,
                         "Sınır Kestirimi":bounds})


def approximate_q3(data, item_table, theta):
    """Yen's Q3-like residual correlations conditional on point theta estimates.

    Exploratory diagnostic, not a calibrated hypothesis test. Estimating theta
    from the same responses typically induces negative residual dependence.
    """
    x = validate_binary(data)
    theta = np.asarray(theta, dtype=float).ravel()
    if len(theta) != len(x):
        raise ValueError("Yetenek sayısı kişi sayısıyla uyuşmuyor.")
    a,b,c,d=(np.asarray(item_table[k],dtype=float) for k in ("a","b","c","d"))
    expected=probabilities(theta,a,b,c,d)
    residual=x-expected
    corr=np.corrcoef(residual,rowvar=False)
    np.fill_diagonal(corr,np.nan)
    names=(list(data.columns) if isinstance(data,pd.DataFrame)
           else [f"M{i+1}" for i in range(x.shape[1])])
    return pd.DataFrame(corr,index=names,columns=names)


def approximate_parameter_uncertainty(data, item_table, model, points=31,
                                      max_parameters=120, a_upper=4.0,
                                      d_lower=0.65):
    """Exploratory OPG covariance based on marginal per-person score vectors.

    Not an observed-Hessian or Louis information SE. Normal intervals can be
    unreliable close to parameter boundaries; return unavailable in that case.
    """
    x=validate_binary(data)
    n,k=x.shape
    active={"1PL":["b"],"2PL":["a","b"],"3PL":["a","b","c"],
            "4PL":["a","b","c","d"]}[model]
    count=k*len(active)
    result=pd.DataFrame({"Madde":list(item_table["Madde"])})
    for name in ("a","b","c","d"):
        result[f"SE({name})"]=np.nan
        result[f"{name} %95 alt"]=np.nan
        result[f"{name} %95 üst"]=np.nan
    reason=structural_identifiability(k,model)["reason"]
    if reason:
        pass
    elif count>max_parameters or n <= 5*count:
        reason="OPG için parametre sayısı fazla veya örneklem yetersiz."
    else:
        a,b,c,d=[np.asarray(item_table[name],dtype=float) for name in ("a","b","c","d")]
        bounds={"a":(0.15,a_upper),"b":(-4.,4.),"c":(0.,.35),"d":(d_lower,1.)}
        for name in active:
            vv=locals()[name]
            lo,hi=bounds[name]
            if np.any((vv-lo)<BOUNDARY_TOLERANCE) or np.any((hi-vv)<BOUNDARY_TOLERANCE):
                reason="Bir veya daha fazla parametre optimizasyon sınırında."
                break
    if not reason:
        gh,w=hermgauss(points)
        nodes=np.sqrt(2)*gh
        z=expit(nodes[:,None]*a[None,:]-(a*b)[None,:])
        probs=np.clip(c[None,:]+(d-c)[None,:]*z,1e-9,1-1e-9)
        lpost=x@np.log(probs).T+(1-x)@np.log1p(-probs).T+np.log(w/np.sqrt(np.pi))[None,:]
        posterior=np.exp(lpost-logsumexp(lpost,axis=1,keepdims=True))
        scores=np.empty((n,count))
        for j in range(k):
            shared=z[:,j]*(1-z[:,j])*(d[j]-c[j])
            derivatives={
                "a":shared*(nodes-b[j]),
                "b":-shared*a[j],
                "c":1-z[:,j],
                "d":z[:,j],
            }
            scale=(x[:,j,None]-probs[None,:,j]) / (
                probs[None,:,j]*(1-probs[None,:,j]))
            for h,name in enumerate(active):
                scores[:,j*len(active)+h]=(
                    posterior*(scale*derivatives[name][None,:])).sum(axis=1)
        opg=scores.T@scores
        try:
            cond=np.linalg.cond(opg)
            if not np.isfinite(cond) or cond>1e9:
                reason="OPG bilgi matrisi kötü koşullanmış/tekil."
            else:
                covariance=np.linalg.inv(opg)
                diag=np.diag(covariance)
                if np.any(diag<=0) or not np.isfinite(diag).all():
                    reason="Pozitif ve sonlu standart hata elde edilemedi."
                else:
                    ses=np.sqrt(diag).reshape(k,len(active))
                    for h,name in enumerate(active):
                        v=np.asarray(item_table[name],dtype=float)
                        se=ses[:,h]
                        result[f"SE({name})"]=se
                        result[f"{name} %95 alt"]=v-1.96*se
                        result[f"{name} %95 üst"]=v+1.96*se
        except np.linalg.LinAlgError:
            reason="OPG bilgi matrisi terslenemedi."
    return result, reason


def item_fit_by_restscore(data, item_table, min_group=5):
    """Exploratory item-rest-score Pearson fit diagnostic; NOT Orlando-Thissen S-X2.

    For each item, group examinees by their observed rest score, compute
    model-expected correct responses averaged over EAP posterior conditional
    on the rest items alone, then calculate Pearson discrepancies. No
    inferential p-value is provided because parameters and groups are fitted.
    """
    x=validate_binary(data)
    n,k=x.shape
    a,b,c,d=[np.asarray(item_table[key],dtype=float) for key in ("a","b","c","d")]
    gh,w=hermgauss(31)
    theta=np.sqrt(2)*gh
    prior=np.log(w/np.sqrt(np.pi))
    p=np.clip(probabilities(theta,a,b,c,d),1e-9,1-1e-9)
    names=list(data.columns) if isinstance(data,pd.DataFrame) else [f"M{i+1}" for i in range(k)]
    out=[]
    total=x.sum(axis=1)
    for j in range(k):
        rest=total-x[:,j]
        other=np.arange(k)!=j
        logpost=(x[:,other]@np.log(p[:,other]).T+
                 (1-x[:,other])@np.log1p(-p[:,other]).T+prior[None,:])
        weights=np.exp(logpost-logsumexp(logpost,axis=1,keepdims=True))
        expected=weights@p[:,j]
        chi=0.
        groups=0
        for score in np.unique(rest):
            take=rest==score
            size=int(take.sum())
            if size<min_group:
                continue
            obs=float(x[take,j].sum())
            exp=float(expected[take].sum())
            variance=max(exp*(1-exp/size),1e-8)
            chi+=(obs-exp)**2/variance
            groups+=1
        out.append({"Madde":names[j],"Rest-puan Pearson":chi if groups>=2 else np.nan,
                    "Kullanılan grup":groups,
                    "Kapsanan kişi":int(sum((rest==v).sum() for v in np.unique(rest)
                                      if (rest==v).sum()>=min_group))})
    return pd.DataFrame(out)


def sx2_pilot(data, item_table, model, points=31, min_expected=5):
    """Experimental Orlando–Thissen-style S-X2 grouped by observed total score.

    Conditional P(Y_i=1|S=s) is calculated from the marginal IRT model,
    integrating over standard normal theta by Gauss-Hermite quadrature and
    Poisson-binomial recursion for the other items. Sparse score groups are
    excluded; no p-value is reported when asymptotic criteria fail. This
    implementation has not been benchmarked against mirt::itemfit.
    """
    from scipy.stats import chi2
    x=validate_binary(data)
    n,k=x.shape
    a,b,c,d=[np.asarray(item_table[q],dtype=float) for q in ("a","b","c","d")]
    gh,hw=hermgauss(points)
    p=np.clip(probabilities(np.sqrt(2)*gh,a,b,c,d),1e-12,1-1e-12)
    weights=hw/np.sqrt(np.pi)
    totals=x.sum(axis=1).astype(int)
    active={"1PL":1,"2PL":2,"3PL":3,"4PL":4}[model]
    names=list(data.columns) if isinstance(data,pd.DataFrame) else list(item_table["Madde"])
    out=[]
    for j in range(k):
        other=np.delete(p,j,axis=1)
        distribution=np.ones((len(weights),1))
        for h in range(k-1):
            ph=other[:,h,None]
            updated=np.zeros((len(weights),distribution.shape[1]+1))
            updated[:,:-1]+=distribution*(1-ph)
            updated[:,1:]+=distribution*ph
            distribution=updated
        # Model probability for full score s and item success at score s.
        weighted_yes=(weights[:,None]*p[:,j,None]*distribution).sum(axis=0)
        weighted_no=(weights[:,None]*(1-p[:,j,None])*distribution).sum(axis=0)
        ps=np.r_[weighted_no,0.] + np.r_[0.,weighted_yes]
        yes=np.r_[0.,weighted_yes]
        cond=np.divide(yes,ps,out=np.zeros(k+1),where=ps>1e-12)
        stat=0.
        used=0
        covered=0
        for score in range(k+1):
            indices=totals==score
            m=int(indices.sum())
            if not m:continue
            expect=m*cond[score]
            if expect<min_expected or m-expect<min_expected:
                continue
            obs=float(x[indices,j].sum())
            stat+=(obs-expect)**2/expect + ((m-obs)-(m-expect))**2/(m-expect)
            used+=1
            covered+=m
        df=used-active-1
        valid=bool(df>0 and covered>=max(30,2*used) and
                   not bool(item_table.iloc[j].get("Sınır uyarısı",False)))
        out.append({"Madde":names[j],"S-X² (pilot)":stat if valid else np.nan,
                    "sd (yaklaşık)":df if valid else np.nan,
                    "p (yaklaşık)":float(chi2.sf(stat,df)) if valid else np.nan,
                    "Kullanılan puan grubu":used,
                    "Kapsanan kişi":covered,
                    "Durum":"Pilot / bağımsız doğrulama gerekli" if valid
                    else "Yetersiz grup veya sınır parametresi"})
    return pd.DataFrame(out)
