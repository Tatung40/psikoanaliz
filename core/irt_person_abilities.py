"""Posterior EAP person scores from fitted category probabilities (all nine IRT models).

Uses the calibrated model's probability curves; no second calibration and no
raw-score-to-theta shortcut. Gaussian N(0,1) latent prior, finite grid quadrature.
"""
import numpy as np
import pandas as pd

def person_abilities(responses, curves, *, model, identifiers=None):
    x=np.asarray(responses)
    if x.ndim!=2 or not np.isfinite(x.astype(float)).all():
        raise ValueError("Kişi yanıt matrisi sonlu ve iki boyutlu olmalı.")
    n,k=x.shape
    if not np.array_equal(x,np.floor(x)) or np.any(x<0):
        raise ValueError("Kategori kodları 0'dan başlayan tamsayılar olmalı.")
    if curves.empty:
        raise ValueError("Kestirilmiş kategori olasılıkları yok.")
    items=list(dict.fromkeys(curves["Madde"].astype(str)))
    grid=np.sort(curves["Theta"].astype(float).unique())
    if len(items)!=k or len(grid)<21 or not np.isfinite(grid).all():
        raise ValueError("Eğri matrisi boyutu veya theta ızgarası geçersiz.")
    if not np.allclose(np.diff(grid),np.diff(grid)[0]):
        raise ValueError("EAP için eşit aralıklı theta ızgarası gerekli.")
    # Categories in fitted curves are in model coding order; do not sort
    # nominal labels alphabetically (NRM has no ordinal category order).
    categories=list(dict.fromkeys(curves.loc[curves["Madde"]==items[0],"Kategori"].astype(str)))
    C=len(categories)
    if C<2 or np.any(x>=C):
        raise ValueError("Yanıt kodları ile model kategorileri uyuşmuyor.")
    probs=np.empty((k,len(grid),C),dtype=float)
    for j,item in enumerate(items):
        subset=curves[curves["Madde"].astype(str)==item]
        for c,cat in enumerate(categories):
            part=subset[subset["Kategori"].astype(str)==cat].sort_values("Theta")
            if len(part)!=len(grid) or not np.allclose(part["Theta"],grid):
                raise ValueError("Eksik veya yinelenen kategori eğrisi.")
            probs[j,:,c]=part["Olasılık"].to_numpy(float)
    if not np.isfinite(probs).all() or np.any(probs<0) or np.any(probs>1):
        raise ValueError("Geçersiz model olasılıkları.")
    if not np.allclose(probs.sum(axis=2),1,atol=1e-5):
        raise ValueError("Kategori olasılıkları toplamı 1 değil.")
    # Gaussian quadrature via equally spaced grid, log-domain normalization.
    log_prior=-0.5*grid**2
    out=np.empty((n,2),float)
    logp=np.log(np.clip(probs,1e-300,1))
    for i in range(n):
        lp=log_prior.copy()
        for j in range(k):
            lp+=logp[j,:,int(x[i,j])]
        w=np.exp(lp-np.max(lp))
        w/=w.sum()
        mean=float(w@grid)
        sd=float(np.sqrt(w@((grid-mean)**2)))
        out[i]=mean,sd
    if identifiers is None:
        identifiers=np.arange(1,n+1)
    if len(identifiers)!=n:
        raise ValueError("Kişi kimliği sayısı yanıt satırlarıyla uyuşmuyor.")
    return pd.DataFrame({
        "Birey":list(identifiers),
        "Ham puan":x.sum(axis=1).astype(int),
        "Yetenek (θ)":out[:,0],
        "SH (θ)":out[:,1],
        "Yöntem":["EAP"]*n,
        "Model":[model]*n,
    })
