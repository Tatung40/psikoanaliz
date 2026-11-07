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
            stat,p,_,expected=chi2_contingency(tab,correction=False)
            rows.append({"Madde":str(item),"Model":model,"Yöntem":"Ham nominal seçenek dağılımı; DIF DEĞİL",
                         "Etki":float(stat),"p":float(p),"Seyrek hücre":bool((expected<5).any())})
        return _family(pd.DataFrame(rows),"Grup dağılımı farklılığı (DIF kanıtı değil)","Belirgin ham farklılık bulunmadı")
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
                         "Etki":np.nan,"p":np.nan,"Not":"Örtüşen grup-rest puanı katmanı yok"})
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
            "Etki":observed,"p":p})
    return _family(pd.DataFrame(rows),"Koşullu fark incelemesi önerilir","Bu taramada belirgin fark yok")


def _family(frame,flag,none,alpha=.05):
    """Decisions use BH-adjusted p over the items of this screen (one family); raw p is kept for transparency."""
    from core.dif_classical import adjust_pvalues
    frame["p (BH)"]=adjust_pvalues(frame["p"],"BH")
    note=frame.pop("Not") if "Not" in frame else pd.Series([None]*len(frame))
    sparse=frame.get("Seyrek hücre",pd.Series(False,index=frame.index))
    frame["Karar"]=[n if isinstance(n,str) else "Raporlanmadı (p hesaplanamadı)" if not np.isfinite(q) else
                    (flag if q<alpha else none)+(" — seyrek hücre (beklenen < 5); χ² yaklaşımı güvenilir olmayabilir" if sp else "")
                    for n,q,sp in zip(note,frame["p (BH)"],sparse)]
    return frame
