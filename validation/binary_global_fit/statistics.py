"""Global dataset-level gates, no item/person pseudoreplication."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
KINDS=("null","nonmonotone","local_dependence","second_factor")
def cp(k,n,a,upper):return float(proportion_confint(k,n,alpha=2*a,method="beta")[int(upper)]) if n else (1. if upper else 0.)
def summary(rows,model,method,phase):
    rr={k:sorted([r for r in rows if r["model"]==model and r["kind"]==k],key=lambda r:r["r"]) for k in KINDS}
    null=rr["null"];n=len(null)
    p=np.array([r.get("result",{}).get(method,np.nan) if not r.get("error") else np.nan for r in null],float)
    bad=~np.isfinite(p);event=(p<.05)|bad;k=int(event.sum());rate=k/max(n,1)
    mcse=float(np.sqrt(rate*(1-rate)/n)) if n else None
    upper=cp(k,n,.025,True);lower=cp(k,n,.025,False)
    power={}
    for kind in KINDS[1:]:
        pp=np.array([r.get("result",{}).get(method,np.nan) if not r.get("error") else np.nan for r in rr[kind]],float)
        good=np.isfinite(pp);s=int((pp<.05).sum());R=len(pp);low=cp(s,R,.025/6,False)
        power[kind]=dict(R=R,k=s,power=s/max(R,1),lower=low,upper=cp(s,R,.025/6,True),
            MCSE=float(np.sqrt((s/max(R,1))*(1-s/max(R,1))/max(R,1))),refusal=float((~good).mean()) if R else 1.)
    goodrows=[r for r in null if r.get("result") and not r.get("error")]
    lat=[r["result"]["seconds"] for r in goodrows]
    ids=[r["identity"] for r in goodrows]
    gates=dict(size=upper<=.07,nontrivial=lower>=.03,MCSE=mcse is not None and mcse<=.004,
        refusal=float(bad.mean())<=.01 if n else False,convergence=sum(bool(r.get("converged")) for r in null)/max(n,1)>=.99,
        complete=phase=="confirm" and n==6000 and all(v["R"]==600 for v in power.values()))
    claims=[k for k,v in power.items() if all(gates.values()) and v["lower"]>.10 and v["refusal"]<=.01]
    return dict(model=model,method=method,null_R=n,TypeI=rate,upper=upper,lower=lower,MCSE=mcse,refusal=float(bad.mean()) if n else 1.,
        power=power,gates=gates,certified_claims=claims,runtime_median=float(np.median(lat)) if lat else None,
        runtime_P95=float(np.percentile(lat,95)) if lat else None,native_identity_max=max(ids) if ids else None,
        errors={e:sum(r.get("error")==e for r in rows if r["model"]==model) for e in set(r.get("error") for r in rows if r["model"]==model) if e})
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input",required=True);ap.add_argument("--phase",required=True);ap.add_argument("--out",required=True);z=ap.parse_args()
    rows=[json.loads(f.read_text()) for f in Path(z.input).rglob("*.json") if f.name.startswith(("3PL_","4PL_"))]
    assert len({(r["model"],r["kind"],r["r"]) for r in rows})==len(rows)
    methods=["p_chi","p_linear","p_nonlinear","p_QLI"]
    selection=json.loads(Path("validation/binary_global_fit/SELECTION.json").read_text()) if z.phase=="confirm" else None
    report={model:{m:summary(rows,model,m,z.phase) for m in ([selection["methods"][model]] if selection else methods)} for model in ("3PL","4PL")}
    Path(z.out).parent.mkdir(parents=True,exist_ok=True);Path(z.out).write_text(json.dumps(report,indent=1))
    print("GLOBAL_REPORT_JSON "+json.dumps(report),flush=True)
if __name__=="__main__":main()
