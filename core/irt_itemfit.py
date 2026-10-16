"""S-X² for unidimensional dichotomous IRT; mirt-style total-score pooling.

Based on the separately supplied reference implementation and tested against
fixed-parameter R mirt 1.48 fixtures. Assumes latent N(0,1), 61 equally
spaced theta points in [-6,6], and fixed group parameters.
"""
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import chi2, norm


def _sumscore_lik(p):
    q,k=p.shape
    out=np.zeros((k+1,q),dtype=float)
    out[0]=1.
    for j in range(k):
        pj=p[:,j]
        out[1:j+2]=out[1:j+2]*(1-pj)+out[:j+1]*pj
        out[0]*=1-pj
    return out


def _collapse_cells(obs,expected,mincell=1.0):
    """Adjacent pooling corresponding to mirt's dichotomous collapseCells."""
    obs=np.array(obs,dtype=float,copy=True)
    expected=np.array(expected,dtype=float,copy=True)
    keep=obs.sum(axis=1)!=0
    obs,expected=obs[keep],expected[keep]
    if len(obs)<2:
        return obs,expected,"Yetersiz dolu toplam puan grubu"
    drop=[]
    for i in range(len(obs)-1):
        if obs[i].sum()==1:
            drop.append(i)
            obs[i+1]+=obs[i]
            expected[i+1]+=expected[i]
    if drop:
        obs=np.delete(obs,drop,axis=0)
        expected=np.delete(expected,drop,axis=0)
    if len(obs)<2:
        return obs,expected,"Gruplama sonrası tek puan grubu"
    if obs[-1].sum()==1:
        obs[-2]+=obs[-1]
        expected[-2]+=expected[-1]
        obs,expected=obs[:-1],expected[:-1]
    if len(obs)<2:
        return obs,expected,"Gruplama sonrası tek puan grubu"
    if np.mean(expected[:,1]>0)<.05:
        return obs,expected,"Beklenen kategori sütunu yetersiz"
    while len(obs)>1:
        low=(expected<mincell)&(expected!=0)
        if not np.any(low):
            break
        row=int(np.where(low.any(axis=1))[0][0])
        if row==0:
            target=1
        elif row==len(obs)-1:
            target=row-1
        else:
            target=row-1 if obs[row-1].sum()<=obs[row+1].sum() else row+1
        obs[target]+=obs[row]
        expected[target]+=expected[row]
        obs=np.delete(obs,row,axis=0)
        expected=np.delete(expected,row,axis=0)
    expected[expected==0]=np.nan
    if len(obs)<2:
        return obs,expected,"Gruplama sonrası tek puan grubu"
    return obs,expected,""


def sx2_mirt_compatible(data,item_table,model,quadpts=61,mincell=1.0):
    """S-X² aligned with mirt's unidimensional dichotomous implementation.

    Computes with fitted item parameters held fixed; this does NOT verify that
    the Python fit itself is equivalent to an independently fitted mirt model.
    Does not interpret non-converged fit as trustworthy automatically.
    """
    from core.irt import validate_binary
    x=validate_binary(data).astype(int)
    n,k=x.shape
    if len(item_table)!=k:
        raise ValueError("Madde parametresi sayısı veri sütunlarıyla uyuşmuyor.")
    if model not in ("1PL","2PL","3PL","4PL"):
        raise ValueError("S-X² modeli geçersiz.")
    if not (quadpts>=21 and 0<mincell<=5):
        raise ValueError("S-X² hesaplama ayarları geçersiz.")
    a,b,c,d=[np.asarray(item_table[name],dtype=float) for name in ("a","b","c","d")]
    if not all(np.isfinite(v).all() for v in (a,b,c,d)):
        raise ValueError("Geçersiz madde parametresi.")
    theta=np.linspace(-6.,6.,quadpts)
    prior=norm.pdf(theta)
    prior/=prior.sum()
    probs=np.clip(c[None,:]+(d-c)[None,:]*expit(
        theta[:,None]*a[None,:]-(a*b)[None,:]),1e-14,1-1e-14)
    total_probs=_sumscore_lik(probs)@prior
    if np.any(total_probs[1:k]<=1e-300):
        raise ValueError("Toplam puan dağılımında sayısal alt taşma.")
    scores=x.sum(axis=1)
    item_par={"1PL":1,"2PL":2,"3PL":3,"4PL":4}[model]
    names=list(data.columns) if isinstance(data,pd.DataFrame) else list(item_table["Madde"])
    results=[]
    for j in range(k):
        other=_sumscore_lik(np.delete(probs,j,axis=1))
        expected=np.column_stack((
            other[:k]@((1-probs[:,j])*prior)/total_probs[:k],
            other[:k]@(probs[:,j]*prior)/total_probs[1:k+1]
        ))
        expected=expected[1:k]
        obs=np.zeros((k+1,2),dtype=float)
        np.add.at(obs,(scores,x[:,j]),1)
        obs=obs[1:k]
        expected*=obs.sum(axis=1)[:,None]
        obs,expected,reason=_collapse_cells(obs,expected,mincell)
        groups=len(obs)
        df=int(np.count_nonzero(~np.isnan(expected))-groups-item_par)
        if reason or df<=0:
            stat=np.nan
            df=max(df,0)
            pvalue=np.nan
            rmsea=np.nan
            if not reason:reason="Serbestlik derecesi yetersiz"
        else:
            with np.errstate(divide="ignore",invalid="ignore"):
                stat=float(np.nansum((obs-expected)**2/expected))
            pvalue=float(chi2.sf(stat,df))
            rmsea=float(np.sqrt(max(stat/df-1,0)/(n-1)))
        results.append({"Madde":names[j],"S-X²":stat,"sd":df,
                        "p":pvalue,"RMSEA.S-X²":rmsea,
                        "Puan grubu":groups,
                        "Durum":reason or "Hesaplandı"})
    return pd.DataFrame(results)
