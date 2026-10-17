"""Paired comparison of single versus multiple EM initializations.

Uses identical simulated data for each strategy. Not a replacement for an
independent mirt reference calibration or a large replication study.
"""
from time import perf_counter
import numpy as np
import pandas as pd
from core.irt_simulation import simulate_irt
from core.irt_multistart import fit_irt_multistart

def compare_starts(true_items,model="3PL",sample_sizes=(200,500),
                   repetitions=3,seed=2026,points=21,max_iter=70):
    if model not in ("3PL","4PL"):
        raise ValueError("Model 3PL veya 4PL olmalıdır.")
    if not 1<=repetitions<=100:
        raise ValueError("Tekrar sayısı geçersiz.")
    seeds=np.random.SeedSequence(seed).spawn(len(sample_sizes)*repetitions)
    rows=[]
    for (n,rep),ss in zip(
        ((n,r+1) for n in sample_sizes for r in range(repetitions)),seeds):
        drawseed=int(ss.generate_state(1)[0])
        data,_=simulate_irt(n,true_items,seed=drawseed)
        for label,starts in (("Tek başlangıç",(0.,)),
                             ("Üç başlangıç",(0.,-1.,1.))):
            begin=perf_counter()
            try:
                run=fit_irt_multistart(data,model=model,starts=starts,
                                        points=points,max_iter=max_iter)
                fit=run["fit"]
                actual=pd.DataFrame(true_items)
                error=np.concatenate([
                    fit["items"][q].to_numpy(float)-actual[q].to_numpy(float)
                    for q in (("a","b","c") if model=="3PL"
                              else ("a","b","c","d"))])
                rows.append({"Model":model,"N":n,"Tekrar":rep,
                             "Tohum":drawseed,"Strateji":label,
                             "Yakınsadı":bool(fit["converged"]),
                             "logLik":fit["loglik"],
                             "Sınır parametresi":len(run["boundary"]),
                             "MAE":float(np.mean(np.abs(error))),
                             "RMSE":float(np.sqrt(np.mean(error**2))),
                             "Süre (sn)":perf_counter()-begin,
                             "Hata":""})
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                rows.append({"Model":model,"N":n,"Tekrar":rep,
                             "Tohum":drawseed,"Strateji":label,
                             "Yakınsadı":False,"logLik":np.nan,
                             "Sınır parametresi":np.nan,
                             "MAE":np.nan,"RMSE":np.nan,
                             "Süre (sn)":perf_counter()-begin,
                             "Hata":str(exc)})
    trials=pd.DataFrame(rows)
    summary=[]
    for (n,label),group in trials.groupby(["N","Strateji"],sort=False):
        good=group[group["Yakınsadı"]]
        summary.append({"Model":model,"N":n,"Strateji":label,
                        "Tekrar":len(group),"Yakınsayan":len(good),
                        "Yakınsama oranı":len(good)/len(group),
                        "MAE (yakınsayan)":good["MAE"].mean(),
                        "RMSE (yakınsayan)":good["RMSE"].mean(),
                        "Sınır parametresi ort.":good["Sınır parametresi"].mean(),
                        "Süre ort. (sn)":group["Süre (sn)"].mean()})
    return {"runs":trials,"summary":pd.DataFrame(summary)}
