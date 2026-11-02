"""Lord chi-square: independent public-library fits + anchor-uncertain delta link.

Physical logistic a,b; theta_R=A theta_F+B, a_F*=a_F/A, b_F*=A b_F+B.
1PL holds a=1 and A=1. Separate samples are independent; within-fit off-diagonals
and every anchor's contribution to A/B remain in the complete Jacobians.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2
from core.irt_python_mirt_adapter import fit_python_mirt
from core.irt_dif import _checked_covariance


def link_contrast(r,f,anchors,item,model):
    r,f=np.asarray(r,float),np.asarray(f,float)
    if model not in ('1PL','2PL'):raise ValueError('Lord DIF yalnız 1PL/2PL')
    k=len(r) if model=='1PL' else len(r)//2
    if r.shape!=f.shape or r.ndim!=1 or len(r)!=(k if model=='1PL' else 2*k) or not np.isfinite([r,f]).all():
        raise ValueError('Geçersiz fiziksel parametre vektörü')
    anchors=np.asarray(anchors,int)
    if len(np.unique(anchors))<2 or len(np.unique(anchors))!=len(anchors) or np.any((anchors<0)|(anchors>=k)):
        raise ValueError('En az iki geçerli çapa gerekli')
    offset=0 if model=='1PL' else k;ix=anchors+offset
    br,bf=r[ix],f[ix];mr,mf=br.mean(),bf.mean()
    dAr=np.zeros(len(r));dAf=np.zeros(len(f));dMr=np.zeros(len(r));dMf=np.zeros(len(f))
    dMr[ix]=1/len(ix);dMf[ix]=1/len(ix)
    A=1.
    if model=='2PL':
        sr=((br-mr)**2).sum();sf=((bf-mf)**2).sum()
        if sr<=0 or sf<=0:raise ValueError('Çapa güçlük yayılımı sıfır; mean/sigma tanımsız')
        A=float(np.sqrt(sr/sf));dAr[ix]=A*(br-mr)/sr;dAf[ix]=-A*(bf-mf)/sf
    B=float(mr-A*mf)
    if not np.isfinite([A,B]).all() or A<=0:raise ValueError('Geçersiz eşitleme A/B')
    Jr=np.zeros((1 if model=='1PL' else 2,len(r)));Jf=np.zeros_like(Jr)
    last=len(Jr)-1
    Jr[last]=(f[item+offset]-mf)*dAr+dMr
    Jf[last]=(f[item+offset]-mf)*dAf-A*dMf
    Jr[last,item+offset]-=1;Jf[last,item+offset]+=A
    delta=[A*f[item+offset]+B-r[item+offset]]
    if model=='2PL':
        Jr[0]=-f[item]/A**2*dAr;Jr[0,item]-=1
        Jf[0]=-f[item]/A**2*dAf;Jf[0,item]+=1/A
        delta.insert(0,f[item]/A-r[item])
    return np.asarray(delta),Jr,Jf,A,B


def physical_fit(fit,model):
    """Canonical free a/b vector and covariance from real Oakes label ordering."""
    if model not in ('1PL','2PL'):raise ValueError('Yalnız 1PL/2PL')
    if not fit.converged:raise ValueError('Tek grup kestirimi yakınsamadı')
    coef=fit.coef();names=list(map(str,coef.index));a=coef.discrimination.to_numpy(float);b=coef.difficulty.to_numpy(float)
    from mirt.estimation.base import _ITEM_PARAMETER_BOUNDS as bounds
    for key,values in [('difficulty',b)]+([('discrimination',a)] if model=='2PL' else []):
        lo,hi=bounds[key]
        if not np.isfinite(values).all() or np.any((values-lo<=1e-6)|(hi-values<=1e-6)):
            raise ValueError('Sınırda veya geçersiz tek-grup parametresi')
    if model=='1PL' and not np.array_equal(a,np.ones(len(a))):raise ValueError('1PL a=1 olmalı')
    labels=list(fit.vcov_labels() if callable(fit.vcov_labels) else fit.vcov_labels)
    wanted=([f'discrimination[{n}]' for n in names] if model=='2PL' else [])+[f'difficulty[{n}]' for n in names]
    if len(labels)!=len(set(labels)) or set(labels)!=set(wanted):raise ValueError('Oakes kovaryans etiketleri uyuşmuyor')
    V=_checked_covariance(fit.vcov() if callable(fit.vcov) else fit.vcov)
    if V.shape!=(len(labels),len(labels)):raise ValueError('Oakes kovaryans boyutu uyuşmuyor')
    order=[labels.index(n) for n in wanted]
    return (b if model=='1PL' else np.r_[a,b]),_checked_covariance(V[np.ix_(order,order)])


def lord_from_parameters(r,f,Vr,Vf,*,anchors,studied,items,model):
    if model not in ('1PL','2PL'):raise ValueError('Lord DIF yalnız 1PL/2PL')
    rows=[];covariances=[]
    for item in studied:
        row=dict(Item=item,Method='Lord χ²',Statistic=np.nan,df=1 if model=='1PL' else 2,p=np.nan,
                 Status='IMPLEMENTED / NOT CERTIFIED',Reason='')
        try:
            vr,vf=_checked_covariance(Vr),_checked_covariance(Vf)
            delta,Jr,Jf,A,B=link_contrast(r,f,[items.index(a) for a in anchors],items.index(item),model)
            if vr.shape!=(len(r),len(r)) or vf.shape!=(len(f),len(f)):raise ValueError('Grup kovaryans boyutu uyuşmuyor')
            cr=Jr@vr@Jr.T;cf=Jf@vf@Jf.T;V=_checked_covariance(cr+cf)
            stat=float(delta@np.linalg.solve(V,delta));row.update(Statistic=stat,p=float(chi2.sf(stat,len(delta))),A=A,B=B)
            for j,kind in enumerate(('b',) if model=='1PL' else ('a','b')):
                row['Delta_'+kind]=float(delta[j]);row['SE_'+kind]=float(np.sqrt(V[j,j]))
            covariances.extend(dict(Item=item,Row=j,Column=k,Covariance=float(V[j,k]),
                                   Reference_contribution=float(cr[j,k]),Focal_contribution=float(cf[j,k]))
                               for j in range(len(delta)) for k in range(len(delta)))
        except (ValueError,TypeError,KeyError,np.linalg.LinAlgError) as exc:
            row.update(Status='UNESTIMABLE / NOT CERTIFIED',Reason=str(exc))
        rows.append(row)
    return dict(table=pd.DataFrame(rows),covariance=pd.DataFrame(covariances))


def lord_dif(data,group,reference,focal,*,model='2PL',anchors,studied,n_quadpts=61,tol=1e-8,max_iter=3000):
    from core.dif_classical import _prepare
    items=list(data.columns);anchors=list(anchors);studied=list(studied)
    if model not in ('1PL','2PL'):raise ValueError('Lord DIF yalnız 1PL/2PL')
    if (len(set(items))!=len(items) or len(set(anchors))<2 or len(set(anchors))!=len(anchors)
            or not set(anchors)<=set(items) or not studied or len(set(studied))!=len(studied)
            or not set(studied)<=set(items)-set(anchors)):
        raise ValueError('En az iki geçerli çapa ve ayrı test maddeleri gerekli')
    x,g,_=_prepare(data,group,reference,focal,False)
    if min(np.bincount(g,minlength=2))<50:raise ValueError('Her grupta en az 50 kişi gerekli')
    fits=[];vectors=[];covariances=[];reason=''
    for gg in (0,1):
        fit=fit_python_mirt(x[g==gg],model=model,n_quadpts=n_quadpts,tol=tol,max_iter=max_iter).fit
        fits.append(fit)
        try:
            v,c=physical_fit(fit,model);vectors.append(v);covariances.append(c)
        except (ValueError,TypeError,AttributeError,np.linalg.LinAlgError) as exc:reason=str(exc)
    if reason:
        return dict(table=pd.DataFrame([dict(Item=i,Method='Lord χ²',Statistic=np.nan,df=1 if model=='1PL' else 2,
            p=np.nan,Status='UNESTIMABLE / NOT CERTIFIED',Reason=reason) for i in studied]),covariance=pd.DataFrame(),fits=fits)
    result=lord_from_parameters(*vectors,*covariances,anchors=anchors,studied=studied,items=items,model=model)
    result.update(fits=fits,vectors=vectors,group_covariances=covariances)
    return result
