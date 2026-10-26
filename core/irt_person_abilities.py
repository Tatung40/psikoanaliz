"""Ordered EAP/posterior SD from calibrated probabilities; never recalibrates."""
import numpy as np
import pandas as pd
from scipy.special import logsumexp


def person_abilities(responses, curves, *, model, identifiers=None, item_names=None,
                     category_labels=None, quadrature=None):
    x=np.asarray(responses,float)
    if x.ndim!=2 or not np.isfinite(x).all():
        raise ValueError('Kişi yanıt matrisi sonlu ve iki boyutlu olmalı.')
    n,k=x.shape
    if not np.array_equal(x,np.floor(x)) or np.any(x<0):
        raise ValueError('Kategori kodları 0’dan başlayan tamsayılar olmalı.')
    if curves.empty:raise ValueError('Kestirilmiş kategori olasılıkları yok.')
    inferred=list(map(str,responses.columns)) if isinstance(responses,pd.DataFrame) else None
    items=list(map(str,item_names)) if item_names is not None else inferred
    actual=list(dict.fromkeys(curves.Madde.astype(str)))
    items=actual if items is None else items
    if inferred is not None and inferred!=items:raise ValueError('Yanıt madde sırası uyuşmuyor.')
    grid=np.sort(curves.Theta.astype(float).unique())
    if len(items)!=k or len(set(items))!=k or set(items)!=set(actual) or len(grid)<21 or not np.isfinite(grid).all():
        raise ValueError('Eğri matrisi boyutu veya theta ızgarası geçersiz.')
    categories=list(map(str,category_labels)) if category_labels is not None else None
    observed=set(curves.Kategori.astype(str))
    if categories is None:
        categories=[str(c) for c in range(len(observed))]
        if observed!=set(categories):raise ValueError('Nominal kategori kod sırası açıkça belirtilmeli.')
    C=len(categories)
    if len(set(categories))!=C or observed!=set(categories) or C<2 or np.any(x>=C):
        raise ValueError('Yanıt kodları ile model kategorileri uyuşmuyor.')
    probs=np.empty((k,len(grid),C),float)
    for j,item in enumerate(items):
        subset=curves[curves.Madde.astype(str)==item]
        for c,cat in enumerate(categories):
            part=subset[subset.Kategori.astype(str)==cat].sort_values('Theta')
            if len(part)!=len(grid) or not np.array_equal(part.Theta.to_numpy(float),grid):
                raise ValueError('Eksik veya yinelenen kategori eğrisi.')
            probs[j,:,c]=part.Olasılık.to_numpy(float)
    if not np.isfinite(probs).all() or np.any(probs<0) or np.any(probs>1):
        raise ValueError('Geçersiz model olasılıkları.')
    if not np.allclose(probs.sum(2),1,rtol=0,atol=1e-6):
        raise ValueError('Kategori olasılıkları toplamı 1 değil.')
    if quadrature is None:
        if not np.allclose(np.diff(grid),np.diff(grid)[0]):
            raise ValueError('EAP için eşit aralıklı ızgara veya açık quadrature gerekli.')
        log_prior=-.5*grid**2
    else:
        nodes,weights=map(lambda v:np.asarray(v,float),quadrature)
        if not np.array_equal(nodes,grid) or weights.shape!=grid.shape or not np.isfinite(weights).all() or np.any(weights<0) or weights.sum()<=0:
            raise ValueError('Quadrature düğüm/ağırlıkları geçersiz.')
        log_prior=np.log(weights/weights.sum(),out=np.full_like(weights,-np.inf),where=weights>0)
    if identifiers is None:identifiers=np.arange(1,n+1)
    ids=pd.Series(list(identifiers))
    if len(ids)!=n or ids.isna().any() or ids.duplicated().any():
        raise ValueError('Kişi kimlikleri satır sayısıyla uyumlu, eksiksiz ve benzersiz olmalı.')
    logp=np.log(probs,out=np.full_like(probs,-np.inf),where=probs>0)
    out=np.empty((n,2),float)
    for i in range(n):
        lp=log_prior.copy()
        for j in range(k):lp+=logp[j,:,int(x[i,j])]
        normalizer=logsumexp(lp)
        if not np.isfinite(normalizer):raise ValueError('Yanıt örüntüsünün model olasılığı sıfır.')
        w=np.exp(lp-normalizer);mean=float(w@grid)
        out[i]=mean,np.sqrt(w@((grid-mean)**2))
    return pd.DataFrame({'Birey':ids.to_numpy(),'Ham puan':x.sum(1).astype(int) if model!='NRM' else pd.NA,
        'Yetenek (θ)':out[:,0],'SH (θ)':out[:,1],'Yöntem':'EAP','Model':model})


def library_person_abilities(fit, responses, *, model, identifiers=None):
    """Use the supported mirt scoring API; nominal category codes are not scores."""
    import mirt
    x=np.asarray(responses,float)
    if not fit.converged or x.ndim!=2 or not np.isfinite(x).all() or np.any(x<0) or not np.array_equal(x,np.floor(x)):
        raise ValueError('Yakınsamış model ve geçerli kategori kodları gerekli.')
    ids=pd.Series(list(identifiers) if identifiers is not None else np.arange(1,len(x)+1))
    if len(ids)!=len(x) or ids.isna().any() or ids.duplicated().any():
        raise ValueError('Kişi kimlikleri eksiksiz ve benzersiz olmalı.')
    scores=mirt.fscores(fit,x.astype(int),method='EAP',n_quadpts=61,person_ids=ids.tolist())
    theta=np.asarray(scores.theta).reshape(-1);sd=np.asarray(scores.standard_error).reshape(-1)
    if not np.array_equal(scores.person_ids,ids) or theta.shape!=(len(x),) or sd.shape!=theta.shape or not np.isfinite(np.r_[theta,sd]).all() or np.any(sd<=0):
        raise ValueError('Kütüphane kişi kestirimleri veya kimlik sırası geçersiz.')
    return pd.DataFrame({'Birey':ids.to_numpy(),'Ham puan':pd.NA if model=='NRM' else x.sum(1),
        'Yetenek (θ)':theta,'SH (θ)':sd,'Yöntem':'EAP','Model':model})
