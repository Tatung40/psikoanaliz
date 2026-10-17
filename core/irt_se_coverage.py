"""Simulation-based interval availability and empirical coverage for 3PL/4PL.

Coverage is only computed for converged fits whose observed-information
matrix is positive definite and estimable away from parameter boundaries.
Always report denominator and refusal rate, which may be highly selective.
"""
import numpy as np
import pandas as pd
from core.irt_simulation import simulate_irt
from core.irt_multistart import fit_irt_multistart
from core.irt_hessian import observed_information_se

_ACTIVE={"3PL":("a","b","c"),"4PL":("a","b","c","d")}

def interval_coverage_study(true_items,model,sample_sizes=(200,),
                            repetitions=3,seed=2026,starts=(0.,-1.,1.),
                            points=31,max_iter=70):
    if model not in _ACTIVE:
        raise ValueError("Only 3PL and 4PL are supported.")
    if not isinstance(repetitions,int) or not 1<=repetitions<=100:
        raise ValueError("Invalid replication count.")
    sizes=tuple(sample_sizes)
    if not sizes or len(sizes)>6 or any(n<30 for n in sizes):
        raise ValueError("Invalid sample sizes.")
    truth=pd.DataFrame(true_items).reset_index(drop=True)
    if len(truth)<3:
        raise ValueError("At least three items required.")
    seeds=np.random.SeedSequence(int(seed)).spawn(len(sizes)*repetitions)
    rows=[]; detail=[]
    for (n,rep),ss in zip(
        ((n,r+1) for n in sizes for r in range(repetitions)),seeds):
        drawseed=int(ss.generate_state(1)[0])
        info={"Model":model,"N":int(n),"Tekrar":rep,"Tohum":drawseed}
        try:
            data,_=simulate_irt(n,truth,seed=drawseed)
            audit=fit_irt_multistart(data,model=model,starts=starts,
                                     points=points,max_iter=max_iter)
            fit=audit["fit"]
            if not fit["converged"]:
                rows.append({**info,"Durum":"Yakınsamadı","SH mümkün":False,
                             "Neden":"EM yakınsamadı"})
                continue
            tab,reason=observed_information_se(
                data,fit["items"],model,points=points,
                a_upper=fit["a_upper"],d_lower=fit["d_lower"])
            if reason:
                rows.append({**info,"Durum":"SH reddedildi","SH mümkün":False,
                             "Neden":reason})
                continue
            rows.append({**info,"Durum":"SH hesaplandı","SH mümkün":True,
                         "Neden":""})
            for parameter in _ACTIVE[model]:
                values=truth[parameter].to_numpy(float)
                lower=tab[parameter+" %95 alt"].to_numpy(float)
                upper=tab[parameter+" %95 üst"].to_numpy(float)
                for j in range(len(truth)):
                    detail.append({**info,"Madde":truth.at[j,"Madde"],
                                   "Parametre":parameter,
                                   "Kapsadı":bool(lower[j]<=values[j]<=upper[j]),
                                   "Aralık genişliği":float(upper[j]-lower[j])})
        except (RuntimeError,ValueError,np.linalg.LinAlgError) as exc:
            rows.append({**info,"Durum":"Hata","SH mümkün":False,
                         "Neden":str(exc)})
    trials=pd.DataFrame(rows)
    covered=pd.DataFrame(detail)
    summary=[]
    for n,group in trials.groupby("N",sort=False):
        allowed=group[group["SH mümkün"]]
        row={"Model":model,"N":n,"Tekrar":len(group),
             "SH hesaplanabilen":len(allowed),
             "SH hesaplanabilirlik oranı":len(allowed)/len(group),
             "Yakınsamayan":int((group["Durum"]=="Yakınsamadı").sum()),
             "SH reddedilen":int((group["Durum"]=="SH reddedildi").sum())}
        for param in _ACTIVE[model]:
            part=covered[(covered["N"]==n)&
                         (covered["Parametre"]==param)] if not covered.empty else pd.DataFrame()
            row["Kapsama("+param+")"]=float(part["Kapsadı"].mean()) if len(part) else np.nan
            row["İncelenen aralık("+param+")"]=len(part)
        summary.append(row)
    return {"runs":trials,"intervals":covered,"summary":pd.DataFrame(summary)}
