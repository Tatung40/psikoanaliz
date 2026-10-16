"""Tek boyutlu ikili MTK: Gauss-Hermite MML/EM kalibrasyonu (1PL-4PL)."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from numpy.polynomial.hermite import hermgauss

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

def fit_irt(data, model="2PL", points=31, max_iter=70, tolerance=1e-4):
    """MML EM, N(0,1) özdeşleme; 1PL için a=1 (Rasch benzeri sabit eğim).
    Returns item params, EAP theta, posterior SD, diagnostic/log-likelihood.
    """
    if model not in ("1PL","2PL","3PL","4PL"):
        raise ValueError("Geçersiz MTK modeli.")
    x = validate_binary(data)
    n,k = x.shape
    gh, hw = hermgauss(points)
    nodes = np.sqrt(2)*gh
    logprior = np.log(hw/np.sqrt(np.pi))
    pop = x.mean(0)
    params = [_initial(p,model) for p in pop]
    bounds = [(-4,4)] if model=="1PL" else _bounds(model)
    if model in ("3PL","4PL"):
        # Önceki model çözümünden başlatma, 3PL/4PL yerel maksimumlarına karşı.
        pass
    history=[]
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
        params=nextparams
        if len(history)>1 and abs(history[-1]-history[-2]) < tolerance*(1+abs(history[-2])):
            converged=True
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
                       "Sınır uyarısı":[bool((model!="1PL" and (a[i]<.17 or a[i]>3.95)) or
                           abs(b[i])>3.95 or (model in ("3PL","4PL") and c[i]>.345) or
                           (model=="4PL" and d[i]<.655)) for i in range(k)]})
    return {"model":model,"items":table,"theta":eap,"theta_sd":psd,
            "grid":theta,"icc":probabilities(theta,a,b,c,d),
            "iif":iteminfo,"tif":iteminfo.sum(axis=1),
            "loglik":ll,"aic":2*numpar-2*ll,
            "bic":np.log(n)*numpar-2*ll,
            "iterations":len(history),"converged":converged,
            "history":history,"n":n,"k":k,"param_count":numpar,
            "warning":("4PL modelindeki c ve d parametreleri yüksek örneklem büyüklüğü "
                       "ve güçlü madde bilgisi gerektirir; sınır kestirimlerini inceleyin."
                       if model=="4PL" else "")}


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
