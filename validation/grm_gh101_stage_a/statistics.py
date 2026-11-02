"""Validation-only statistics; no optimizer or production inference changes."""
import itertools
import math
import numpy as np
from scipy.stats import chi2, binomtest
from core.dif_classical import adjust_pvalues
from validation.ordinal_dif_paired.experiment import binomial_ci, classify
from validation.ordinal_dif_impact_zero.experiment import mean_stats, exact_paired_ci

PLANNED=100

def decisions(row):
    if not row.get("valid", False):
        return {"valid":False,"p":None,"holm_event":None,"bh_event":None}
    out={}
    for name in ("omnibus","thresholds","slope"):
        lr=[v for v in row["lr"] if v["test"]==name]
        if len(lr)!=2 or {v["item"] for v in lr}!={"P09","P10"} or not all(v["valid"] for v in lr):
            raise ValueError("Incomplete LR family")
        lr.sort(key=lambda v:v["item"])
        out[name]=classify([float(chi2.sf(v["lr"]["161"],v["df"])) for v in lr],.05)
        out[name]["p_GH101_sensitivity"]=[float(chi2.sf(v["lr"]["101"],v["df"])) for v in lr]
    return {**out["omnibus"],"component_families":out}

def rate(values):
    n=len(values);k=sum(values)
    return dict(n=n,events=k,rate=k/n if n else None,
        mcse=float(np.std(values,ddof=1)/np.sqrt(n)) if n>1 else None,
        cp95=binomial_ci(k,n,.05),cp975=binomial_ci(k,n,.025),
        joint11_cp=binomial_ci(k,n,.05/11),
        unknown_failure_empirical_bounds=[k/PLANNED,(k+PLANNED-n)/PLANNED])

def contrast(quads, interaction=False):
    width=4 if interaction else 2
    values=[a-b-c+d for a,b,c,d in quads] if interaction else [a-b for a,b in quads]
    out=mean_stats(values,width);out["joint11_ci"]=mean_stats(values,width,.05/11)["hoeffding_ci"]
    out["missing_from_fixed100"]=PLANNED-len(values)
    out["unknown_failure_empirical_bounds"]=[(sum(values)-width/2*(PLANNED-len(values)))/PLANNED,
                                             (sum(values)+width/2*(PLANNED-len(values)))/PLANNED]
    if interaction:
        out["quadruple_order"]=["GRM0","ordinal0","GRMm","ordinalm"]
        out["table"]={"".join(map(str,q)):quads.count(q) for q in itertools.product((0,1),repeat=4)}
        def ci(i,j):
            return exact_paired_ci(sum(q[i] and not q[j] for q in quads),sum(q[j] and not q[i] for q in quads),len(quads),.025)
        zero,minus=ci(0,1),ci(2,3)
        out["exact_paired95"]=[max(-2.,zero[0]-minus[1]),min(2.,zero[1]-minus[0])]
        out["mcnemar_p"]=None
    else:
        b=sum(a and not b for a,b in quads);c=sum(b and not a for a,b in quads)
        out["table"]={"".join(map(str,q)):quads.count(q) for q in itertools.product((0,1),repeat=2)}
        out["exact_paired95"]=exact_paired_ci(b,c,len(quads))
        out["mcnemar_p"]=float(binomtest(b,b+c,.5).pvalue) if b+c else (1. if quads else None)
    return out

def summarize(rows, ordinal):
    by={(r["seed"],r["impact"]):r for r in rows}
    if len(by)!=len(rows):raise ValueError("Duplicate attempts")
    conditions={}
    def event(seed,impact,method):
        if method=="GRM":
            r=by.get((seed,impact),{});v=r.get("decisions",{})
        else:v=ordinal.get((seed,impact),{}).get("methods",{}).get(method,{})
        return int(v["holm_event"]) if v.get("valid",False) else None
    for impact in ("minus","zero"):
        cases=[r for r in rows if r["impact"]==impact];valid=[r["decisions"] for r in cases if r.get("valid",False)]
        conditions[impact]=dict(attempted=len(cases),valid=len(valid),invalid=len(cases)-len(valid),
            not_run=PLANNED-len(cases),holm_fwer=rate([int(v["holm_event"]) for v in valid]),
            bh_global_null_fdr=rate([int(v["bh_event"]) for v in valid]),
            raw_type1=dict(n=len(valid),mean=float(np.mean([sum(v["raw_rejections"])/2 for v in valid])) if valid else None,
                mcse=float(np.std([sum(v["raw_rejections"])/2 for v in valid],ddof=1)/np.sqrt(len(valid))) if len(valid)>1 else None))
        conditions[impact]["ordinal"]={m:rate([event(s,impact,m) for s in range(2026110100,2026110200)
                if event(s,impact,m) is not None]) for m in ("total","rest")}
    specs=[]
    for impact in ("minus","zero"):
        for method in ("total","rest"):
            specs.append((f"GRM-minus-{method}_{impact}",[(impact,"GRM"),(impact,method)]))
    specs.append(("GRM_zero-minus-minus",[("zero","GRM"),("minus","GRM")]))
    for impact in ("minus","zero"):specs.append((f"total-minus-rest_{impact}",[(impact,"total"),(impact,"rest")]))
    for method in ("total","rest"):specs.append((f"(GRM-{method})_zero-minus-minus",
        [("zero","GRM"),("zero",method),("minus","GRM"),("minus",method)]))
    out={}
    for name,selectors in specs:
        pairs=[]
        for seed in range(2026110100,2026110200):
            values=tuple(event(seed,*select) for select in selectors)
            if all(v is not None for v in values):pairs.append(values)
        out[name]=contrast(pairs,len(selectors)==4)
    binary=[name for name,sel in specs if len(sel)==2]
    p=[out[n]["mcnemar_p"] if out[n]["mcnemar_p"] is not None else 1. for n in binary]
    for method in ("Holm","BH"):
        for name,value in zip(binary,adjust_pvalues(p,method)):
            out[name][method+"_seven_p"]=float(value) if out[name]["n"] else None
    return dict(scientific_status="STAGE A / NOT CERTIFIED",planned_analyses=200,attempted=len(rows),
        valid=sum(r.get("valid",False) for r in rows),invalid=sum(not r.get("valid",False) for r in rows),
        not_run=200-len(rows),complete_valid_pairs=sum(all(by.get((s,k),{}).get("valid",False) for k in ("minus","zero")) for s in range(2026110100,2026110200)),
        conditions=conditions,contrasts=out,inference="Conditional descriptions only; no scientific certification, no 500/2500 stages, no GH61 pooling.")
