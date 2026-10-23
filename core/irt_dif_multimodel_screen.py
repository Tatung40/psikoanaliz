"""Model-agnostic, observed-score DIF screening; NOT IRT likelihood DIF.

Binary and ordinal: conditional item mean contrasts within other-item rest-score
strata. Permutation p-values shuffle group labels only within strata.
NRM: report unconditional option-distribution disparity only, explicitly not DIF.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

MODELS=("Rasch","1PL","2PL","3PL","4PL","GRM","PCM","GPCM","RSM","NRM")

def screen_mtk_groups(data,items,group,reference,focal,model,*,permutations=199,seed=42):
    if model not in MODELS:raise ValueError("Unsupported model")
    frame=pd.DataFrame(data)[list(items)].copy()
    if len(frame)<40 or len(items)<3:raise ValueError("At least 40 respondents and three items needed")
    groups=pd.Series(group,index=frame.index)
    mask=groups.isin([reference,focal])
    x=frame.loc[mask];g=(groups.loc[mask]==focal).to_numpy(int)
    if (g==0).sum()<20 or (g==1).sum()<20:raise ValueError("At least 20 respondents per group")
    if x.isna().any().any():raise ValueError("Complete item responses required")
    rng=np.random.default_rng(seed);rows=[]
    if model=="NRM":
        for item in x:
            tab=pd.crosstab(g,x[item])
            tab=tab.reindex(index=[0,1],fill_value=0)
            if tab.shape[1]<2:raise ValueError("Constant nominal item")
            stat,p=chi2_contingency(tab,correction=False)[:2]
            rows.append({"Madde":str(item),"Model":model,"Yöntem":"Ham nominal seçenek dağılımı; DIF DEĞİL",
                         "Etki":float(stat),"p":float(p),"Karar":"Grup dağılımı farklılığı (DIF kanıtı değil)" if p<.05 else "Belirgin ham farklılık bulunmadı"})
        return pd.DataFrame(rows)
    numeric=x.apply(pd.to_numeric,errors="raise").to_numpy(float)
    if not np.isfinite(numeric).all():raise ValueError("Nonfinite responses")
    if model in ("Rasch","1PL","2PL","3PL","4PL") and not np.isin(numeric,[0,1]).all():
        raise ValueError("Dichotomous MTK models require 0/1")
    if model in ("GRM","PCM","GPCM","RSM") and (not np.array_equal(numeric,np.floor(numeric)) or (numeric<0).any()):
        raise ValueError("Ordinal responses must be nonnegative integer scores")
    for j,item in enumerate(items):
        y=numeric[:,j];rest=numeric.sum(axis=1)-y
        # Exact score strata, no arbitrary theta or score scale from nominal labels.
        strata=[np.flatnonzero(rest==score) for score in np.unique(rest)]
        eligible=[ix for ix in strata if len(np.unique(g[ix]))==2 and len(ix)>=4]
        if not eligible:
            rows.append({"Madde":str(item),"Model":model,"Yöntem":"Rest puanı koşullu permütasyon taraması; MTK DIF DEĞİL",
                         "Etki":np.nan,"p":np.nan,"Karar":"Örtüşen grup-rest puanı katmanı yok"})
            continue
        def statistic(groups):
            vals=[];weights=[]
            for ix in eligible:
                d=y[ix][groups[ix]==1].mean()-y[ix][groups[ix]==0].mean()
                vals.append(d);weights.append(len(ix))
            return float(np.average(vals,weights=weights))
        observed=statistic(g)
        ge=0
        for _ in range(permutations):
            gp=g.copy()
            for ix in eligible:gp[ix]=rng.permutation(gp[ix])
            if abs(statistic(gp))>=abs(observed)-1e-12:ge+=1
        p=(ge+1)/(permutations+1)
        rows.append({"Madde":str(item),"Model":model,
            "Yöntem":"Rest puanı koşullu permütasyon taraması; MTK DIF DEĞİL",
            "Etki":observed,"p":p,
            "Karar":"Koşullu fark incelemesi önerilir" if p<.05 else "Bu taramada belirgin fark yok"})
    return pd.DataFrame(rows)
