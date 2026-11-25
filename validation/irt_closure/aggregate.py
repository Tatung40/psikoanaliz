"""Independent cohort-level evidence aggregation; datasets, not intervals, are clusters."""
import argparse,json,os
from pathlib import Path
import numpy as np
from scipy.stats import beta,t,norm
from core.irt_binary_inference import Z95
from validation.irt_closure.engine import write_json
from validation.coverage_precision import coverage_precision


def ci(values,family=1):
    x=np.asarray(values,float);n=len(x)
    if n<2:return {"mean":float(x.mean()) if n else None,"MCSE":None,"CI":[None,None]}
    m=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(n));q=float(t.ppf(1-.05/(2*family),n-1))
    return {"mean":m,"MCSE":se,"CI":[m-q*se,m+q*se]}


def cp_upper(k,n,alpha=.05/24):
    return float(beta.ppf(1-alpha,k+1,n-k)) if k<n else 1.


def p3(rows,expected,bootstrap=False):
    byitem=[];worst=[];valid=[];completed=len(rows)
    for r in rows:
        if bootstrap:
            b=r.get("p1") or {};p=np.array(b.get("bootstrap_pvalue",[None]*12),float)
            okay=np.array(b.get("p3_non_evaluable_rate",[1]*12),float)<=.01
        else:
            b=r.get("p3") or {};p=np.array(b.get("pvalue",[None]*12),float)
            okay=np.isfinite(p)&(np.array(b.get("df",[None]*12),float)>0)
        okay &= np.isfinite(p)&(p>=0)&(p<=1)
        valid.append(okay);worst.append((p<=.05)|~okay)
    if not rows:return {"status":"NOT EXECUTED","certified":False}
    worst=np.asarray(worst);valid=np.asarray(valid);rate=worst.mean(0)
    for j in range(12):
        events=int(worst[:,j].sum())
        byitem.append({"item":j+1,"rejections_failures_as_events":events,"R":completed,
            "TypeI_worst_case":float(rate[j]),"upper_CP_simultaneous":cp_upper(events,completed),
            "non_evaluable_rate":float(1-valid[:,j].mean())})
    pooled=ci(worst.mean(1),24)
    gates={"complete":completed==expected,"pooled_MCSE":pooled["MCSE"] is not None and pooled["MCSE"]<=.004,
        "all_item_CP":all(s["upper_CP_simultaneous"]<=.07 for s in byitem),
        "evaluable":bool((1-valid.mean(0)<=.01).all())}
    return {"method":"full-refit bootstrap B99" if bootstrap else "native chi-square",
        "R":completed,"expected":expected,"items":byitem,"pooled":pooled,
        "gates":gates,"certified":all(gates.values()),
        "scope":"Approximate alpha=.05 calibration in preregistered N3000/J12 null DGP; tolerance+.02, not all-alpha p uniformity"}


def p2(rows,expected=300):
    rows=[r for r in rows if r["r"]<expected]
    successful=[r["p2"] for r in rows if r.get("p2")]
    failures=len(rows)-len(successful)
    if not successful:return {"certified":False,"R":len(rows),"valid":0,"status":"NOT EXECUTED / FAILED"}
    family=54
    bias=ci([x["bias"] for x in successful],family)
    efficiency=ci([x["excess_mse_fraction"] for x in successful],family)
    credible=ci([x["credible_F"]["marginal"] for x in successful],family)
    score=ci([x["exact_score_F"]["marginal"] for x in successful],family)
    grid=[ci([x["exact_score_F"]["conditional_theta"][j] for x in successful],family) for j in range(21)]
    tails=[ci([x["exact_score_F"]["tails"][j] for x in successful],family) for j in range(2)]
    fraction=failures/len(rows)
    credible_sensitivity=[(1-fraction)*credible["CI"][0],(1-fraction)*credible["CI"][1]+fraction]
    recovery={"complete":len(rows)==expected,"evaluable":fraction<=.01,
        "bias":bias["CI"][0]>=-.05 and bias["CI"][1]<=.05,
        "efficiency":efficiency["CI"][1]<=.05}
    credible_gate={**recovery,"coverage":credible_sensitivity[0]>=.93 and credible_sensitivity[1]<=.97,
        "MCSE":credible["MCSE"]<=.004}
    exact_gate={**recovery,"marginal":(1-fraction)*score["CI"][0]>=.95,
        "all_grid":all((1-fraction)*s["CI"][0]>=.95 for s in grid),
        "tails":all((1-fraction)*s["CI"][0]>=.95 for s in tails)}
    return {"R":len(rows),"expected":expected,"valid":len(successful),"failure_rate":fraction,
        "bias":bias,"excess_MSE_fraction":efficiency,"credible_marginal":credible,
        "credible_failure_sensitivity":credible_sensitivity,
        "credible_conditional_mean":np.mean([x["credible_F"]["conditional_theta"] for x in successful],axis=0).tolist(),
        "credible_tail_mean":np.mean([x["credible_F"]["tails"] for x in successful],axis=0).tolist(),
        "exact_score_marginal":score,"exact_score_grid":grid,"exact_score_tails":tails,
        "mean_rmse":float(np.mean([x["rmse"] for x in successful])),
        "known_credible_marginal":successful[0]["credible_K"]["marginal"],
        "known_exact_score_conditional":successful[0]["exact_score_K"]["conditional_theta"],
        "recovery_gates":recovery,"credible_gates":credible_gate,"exact_score_gates":exact_gate,
        "recovery_certified":all(recovery.values()),"credible_certified":all(credible_gate.values()),
        "exact_score_certified":all(exact_gate.values())}


def p1(rows,expected):
    rows=[r for r in rows if r["r"]<expected]
    selected=[r for r in rows if (r.get("p1") or {}).get("p1_valid") and r["p1"]["B"]==399]
    out={"R":len(rows),"expected":expected,"eligible":len(selected),"parameters":{},"certified":False}
    if len(selected)<2:return out
    qs=selected[0]["p1"]["qs"]
    convergence=sum(r.get("calibration",{}).get("converged",False) for r in rows)/len(rows)
    refused=1-len(selected)/len(rows)
    gates={"complete":len(rows)==expected,"convergence":convergence>=.99,"refusal":refused<=.01}
    for ix,q in enumerate(qs):
        errors=np.array([r["p1"]["error"][ix::len(qs)] for r in selected])
        covered=np.array([r["p1"]["covered"][ix::len(qs)] for r in selected])
        ses=np.array([r["p1"]["sampling_se"][ix::len(qs)] for r in selected])
        points=np.array([r["p1"]["estimate"][ix::len(qs)] for r in selected])
        precision=coverage_precision([np.column_stack([er,np.zeros(12),cv]) for er,cv in zip(errors,covered)],len(rows),eligible_column=1)
        coverage=float(covered.mean());bias=float(errors.mean());rmse=float(np.sqrt(np.mean(errors**2)))
        centered=points-points.mean(0)
        numer=(ses**2).mean(1)
        denom=(centered**2).mean(1)*len(selected)/(len(selected)-1)
        ratio=np.sqrt(numer.mean()/denom.mean())
        influence=(numer-ratio**2*denom)/(2*ratio*denom.mean())
        se=float(influence.std(ddof=1)/np.sqrt(len(influence)))
        crit=float(t.ppf(1-.05/(2*8),len(influence)-1))
        ratio_ci=[float(ratio-crit*se),float(ratio+crit*se)]
        ratio_limits=[float(norm.ppf(.965)/Z95),float(norm.ppf(.985)/Z95)]
        out["parameters"][q]={"coverage":coverage,**precision,"bias":bias,"RMSE":rmse,
            "sampling_SE_ratio":float(ratio),"sampling_SE_ratio_CI":ratio_ci,"SE_limits":ratio_limits,
            "per_item_coverage":covered.mean(0).tolist(),"refusal_rate":refused,
            "mean_inner_refusal":float(np.mean([r["p1"]["inner_refusal"] for r in selected]))}
        gates[q+"_coverage"]=abs(coverage-.95)<=.02
        gates[q+"_precision"]=precision["coverage_mc_se"] is not None and precision["coverage_mc_se"]<=.004
        gates[q+"_SE"]=ratio_ci[0]>=ratio_limits[0] and ratio_ci[1]<=ratio_limits[1]
        gates[q+"_RMSE"]=rmse<=(.30 if q in "ab" else .07)
        if q in "ab":gates[q+"_bias"]=abs(bias)<=.05
    out["gates"]=gates;out["certified"]=all(gates.values())
    return out


def aggregate(root):
    records={}
    for file in Path(root).rglob("result_*.json"):
        r=json.loads(file.read_text());key=(r["stage"],r["model"],r["r"])
        records[key]=r
    output={"calibration_clusters":len(records),"scope":"N3000/J12, common.truth, independent normal latent; no J30","models":{}}
    for model in ("3PL","4PL"):
        pilot=sorted([r for (s,m,i),r in records.items() if s=="pilot" and m==model],key=lambda r:r["r"])
        baseline=sorted([r for (s,m,i),r in records.items() if s=="baseline" and m==model],key=lambda r:r["r"])
        confirm=sorted([r for (s,m,i),r in records.items() if s=="confirm" and m==model],key=lambda r:r["r"])
        output["models"][model]={"development_P1":p1(pilot,20),"baseline_P3":p3(baseline,3000),
            "confirmation_P2":p2(baseline,300),
            "confirmation_P1":p1(confirm,400),"bootstrap_P3":p3(confirm,1500,True)}
        # Development outcomes are NEVER certified.
        output["models"][model]["development_P1"]["certified"]=False
    return output


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--input",required=True);ap.add_argument("--output",required=True);a=ap.parse_args()
    report=aggregate(a.input)
    phase=json.loads(Path(__file__).with_name("RUN.json").read_text()).get("phase")
    if phase=="archive":
        canonical_path=Path(__file__).resolve().parents[1]/"evidence/irt_closure/REPORT.json"
        canonical=json.loads(canonical_path.read_text())
        def equivalent(x,y):
            if isinstance(x,dict):return isinstance(y,dict) and x.keys()==y.keys() and all(equivalent(x[k],y[k]) for k in x)
            if isinstance(x,list):return isinstance(y,list) and len(x)==len(y) and all(equivalent(a,b) for a,b in zip(x,y))
            if isinstance(x,(int,float)) and not isinstance(x,bool):return isinstance(y,(int,float)) and abs(x-y)<=1e-12
            return x==y
        for model in ("3PL","4PL"):
            for name in ("baseline_P3","confirmation_P2"):
                if not equivalent(report["models"][model][name],canonical["models"][model][name]):
                    raise ValueError("Canonical cohort verification failed: "+model+" "+name)
        canonical["read_only_verified_calibrations"]=report["calibration_clusters"]
        report=canonical
    write_json(a.output,report)
    print("AGGREGATE_JSON "+json.dumps(report))
