"""Reproducible IRT simulation and parameter-recovery diagnostics.

Not a certification of 3PL/4PL calibration accuracy. Recovery must be
assessed across sample sizes, test lengths, true-parameter regimes and seeds.
"""
import numpy as np
import pandas as pd
from core.irt import probabilities, fit_irt
from core.irt_se_diagnostics import hessian_boundary_diagnostics

_MODELS={"1PL":("b",),"2PL":("a","b"),
         "3PL":("a","b","c"),"4PL":("a","b","c","d")}

def simulate_irt(n, true_items, seed=2026):
    """Simulate independent item responses conditional on N(0,1) abilities."""
    if not isinstance(n,(int,np.integer)) or n<30:
        raise ValueError("Örneklem en az 30 kişi olmalıdır.")
    if not isinstance(seed,(int,np.integer)) or seed<0:
        raise ValueError("Tohum negatif olmayan tamsayı olmalıdır.")
    p=pd.DataFrame(true_items)
    if len(p)<3 or p["Madde"].duplicated().any():
        raise ValueError("En az üç farklı madde olmalıdır.")
    a,b,c,d=[p[q].to_numpy(float) for q in ("a","b","c","d")]
    if not (np.isfinite(np.column_stack((a,b,c,d))).all()
            and np.all(a>0) and np.all((c>=0)&(c<d)&(d<=1))):
        raise ValueError("Geçersiz IRT parametreleri.")
    rng=np.random.default_rng(int(seed))
    ability=rng.standard_normal(n)
    pr=probabilities(ability,a,b,c,d)
    responses=pd.DataFrame((rng.random(pr.shape)<pr).astype(int),
                           columns=list(p["Madde"]))
    return responses,ability


def parameter_recovery(data,true_items,model,starts=(0.,),**fit_kwargs):
    """Return fitted model, recovery table and model-level recovery summary.

    Recovery is descriptive, not proof of convergence or identification.
    Fixed parameters are omitted from active recovery metrics.
    """
    if model not in _MODELS:
        raise ValueError("Bilinmeyen MTK modeli.")
    truth=pd.DataFrame(true_items).reset_index(drop=True)
    if list(data.columns)!=list(truth["Madde"]):
        raise ValueError("Gerçek parametrelerle veri madde sırası uyuşmuyor.")
    if model in ("3PL","4PL"):\n        from core.irt_multistart import fit_irt_multistart\n        audited=fit_irt_multistart(data,model=model,starts=starts,**fit_kwargs)\n        result=audited["fit"]\n    else:\n        result=fit_irt(data,model=model,**fit_kwargs)
    fitted=result["items"].reset_index(drop=True)
    diag=hessian_boundary_diagnostics(
        fitted,model,a_upper=result["a_upper"],d_lower=result["d_lower"])
    records=[]
    for param in _MODELS[model]:
        for j in range(len(truth)):
            value=float(truth.at[j,param])
            estimate=float(fitted.at[j,param])
            records.append({"Madde":truth.at[j,"Madde"],
                            "Parametre":param,"Gerçek":value,
                            "Kestirim":estimate,"Hata":estimate-value,
                            "Mutlak hata":abs(estimate-value)})
    recovery=pd.DataFrame(records)
    summary={
        "model":model,"n":result["n"],"k":result["k"],
        "converged":bool(result["converged"]),
        "iterations":int(result["iterations"]),
        "loglik":float(result["loglik"]),\n        "start_count":len(starts),
        "boundary_item_count":int(diag["Madde"].nunique()),
        "boundary_parameter_count":int(len(diag)),
        "MAE":float(recovery["Mutlak hata"].mean()),
        "RMSE":float(np.sqrt(np.mean(recovery["Hata"]**2)))
    }
    for param,part in recovery.groupby("Parametre"):
        summary[f"MAE({param})"]=float(part["Mutlak hata"].mean())
        summary[f"Bias({param})"]=float(part["Hata"].mean())
    return {"fit":result,"recovery":recovery,
            "boundary_diagnostics":diag,"summary":summary}


def repeated_recovery(true_items, models=("3PL","4PL"),
                      sample_sizes=(200,500,1000,2000),
                      repetitions=10, seed=2026, starts=(0.,), **fit_kwargs):
    """Run reproducible, independent replications across sample sizes/models.

    Each simulation draw is reused across requested models for a fair paired
    comparison. Failures are logged and never treated as successful estimates.
    Nonconverged estimates are reported separately from converged recovery.
    Computationally expensive: call explicitly, not on Streamlit page load.
    """
    if not isinstance(repetitions,(int,np.integer)) or not 1<=repetitions<=500:
        raise ValueError("Tekrar sayısı 1–500 aralığında olmalıdır.")
    if not isinstance(seed,(int,np.integer)) or seed<0:
        raise ValueError("Tohum negatif olmayan bir tamsayı olmalıdır.")
    models=tuple(models)
    sizes=tuple(sample_sizes)
    if not models or any(m not in _MODELS for m in models):
        raise ValueError("Geçersiz model listesi.")
    if not sizes or any(not isinstance(n,(int,np.integer)) or n<30 for n in sizes):
        raise ValueError("Örneklem büyüklükleri en az 30 olmalıdır.")
    if not isinstance(true_items,pd.DataFrame):
        true_items=pd.DataFrame(true_items)
    parameters=true_items.copy()
    master=np.random.SeedSequence(int(seed))
    tasks=[(n,r) for n in sizes for r in range(repetitions)]
    children=master.spawn(len(tasks))
    runs=[]
    details=[]
    for (n,rep),child in zip(tasks,children):
        drawseed=int(child.generate_state(1,dtype=np.uint32)[0])
        x,_=simulate_irt(int(n),parameters,seed=drawseed)
        for model in models:
            meta={"Model":model,"N":int(n),"Tekrar":rep+1,"Tohum":drawseed}
            try:
                output=parameter_recovery(x,parameters,model,starts=starts,**fit_kwargs)
                info=output["summary"]
                row={**meta,"Durum":"Yakınsadı" if info["converged"]
                     else "Yakınsamadı","Yakınsama":bool(info["converged"]),
                     "Sınırdaki madde":info["boundary_item_count"],
                     "Sınırdaki parametre":info["boundary_parameter_count"],
                     "MAE":info["MAE"],"RMSE":info["RMSE"],\n                     "Başlangıç sayısı":info["start_count"],"Hata":""}
                for q in _MODELS[model]:
                    row["MAE("+q+")"]=info["MAE("+q+")"]
                    row["Bias("+q+")"]=info["Bias("+q+")"]
                runs.append(row)
                detail=output["recovery"].copy()
                for key,val in meta.items():
                    detail[key]=val
                detail["Yakınsama"]=bool(info["converged"])
                details.append(detail)
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                runs.append({**meta,"Durum":"Hata","Yakınsama":False,
                             "Hata":str(exc)})
    trials=pd.DataFrame(runs)
    full=pd.concat(details,ignore_index=True) if details else pd.DataFrame()
    rows=[]
    for (model,n),group in trials.groupby(["Model","N"],sort=False):
        ok=group[group["Durum"]=="Yakınsadı"]
        row={"Model":model,"N":int(n),"Tekrar":len(group),
             "Yakınsayan":len(ok),"Yakınsama oranı":len(ok)/len(group),
             "Hatalı koşu":int((group["Durum"]=="Hata").sum()),
             "Sınırdaki madde ort.":float(ok["Sınırdaki madde"].mean())
                if len(ok) else np.nan,
             "Sınırdaki parametre ort.":float(ok["Sınırdaki parametre"].mean())
                if len(ok) else np.nan}
        converged=full[(full["Model"]==model)&(full["N"]==n)&full["Yakınsama"]] if not full.empty else pd.DataFrame()
        for param in _MODELS[model]:
            part=converged[converged["Parametre"]==param] if not converged.empty else pd.DataFrame()
            row["Bias("+param+")"]=float(part["Hata"].mean()) if len(part) else np.nan
            row["MAE("+param+")"]=float(part["Mutlak hata"].mean()) if len(part) else np.nan
            row["RMSE("+param+")"]=float(np.sqrt(np.mean(part["Hata"]**2))) if len(part) else np.nan
        rows.append(row)
    return {"summary":pd.DataFrame(rows),"runs":trials,"item_recovery":full}
