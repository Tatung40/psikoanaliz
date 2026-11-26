"""Dataset-cluster statistics; no fits. Never treats a pilot as certification."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy import stats
from validation.p3_sx2_aggregate.aggregate import cp_upper,cp_lower


def summarize(rows,model,phase):
    null=sorted([r for r in rows if r["model"]==model and not r["alt"]],key=lambda r:r["r"])
    alt=sorted([r for r in rows if r["model"]==model and r["alt"]],key=lambda r:r["r"])
    report={}
    for method in ("p_emp","p_standardized"):
        n=len(null);p=np.full((n,12),np.nan)
        lat=[];identity=0
        for i,r in enumerate(null):
            if "result" in r and not r.get("error"):
                items=r["result"]["items"];p[i]=[x[method] for x in items]
                native=np.asarray(r["p_chi"]);p[i,~np.isfinite(native)]=np.nan
                lat.append(r["result"]["seconds"]);identity=max(identity,r["identity"])
        invalid=~np.isfinite(p);reject=(p<.05)|invalid
        cl=reject.mean(1);mcse=float(cl.std(ddof=1)/np.sqrt(n)) if n>1 else float("nan")
        upper=[cp_upper(int(k),n,.05/24) for k in reject.sum(0)] if n else [1.]*12
        pa=np.array([r.get("result",{}).get("items",[{}])[0].get(method,np.nan) if not r.get("error") else np.nan for r in alt])
        detected=int(np.sum(pa<.05));low=cp_lower(detected,len(alt),.025) if len(alt) else 0.
        err={}
        for r in null+alt:
            if r.get("error"):err[r["error"]]=err.get(r["error"],0)+1
        complete=n==6000 and len(alt)==600 and len({r["r"] for r in null})==6000 and len({r["r"] for r in alt})==600
        gates={"CP_upper<=.07":max(upper)<=.07,"cluster_MCSE<=.004":mcse<=.004,
            "pooled95_lower>=.03":float(cl.mean())-stats.norm.ppf(.975)*mcse>=.03 if n else False,
            "nonevaluable<=.01":float(invalid.mean())<=.01 if n else False,"identity<=1e-8":identity<=1e-8,
            "power_CP_lower>.10":low>.10,"complete_independent_confirmation":phase=="confirm" and complete}
        report[method]={"null_R":n,"power_R":len(alt),"typeI":float(cl.mean()) if n else None,"MCSE":mcse,
            "per_item":reject.mean(0).tolist() if n else [],"upper":upper,"power":detected/max(1,len(alt)),"power_lower":low,
            "power_upper":cp_upper(detected,len(alt),.025) if len(alt) else 1.,"refusal":float(invalid.mean()) if n else 1.,
            "power_refusal":float((~np.isfinite(pa)).mean()) if len(pa) else 1.,
            "runtime_median":float(np.median(lat)) if lat else None,"runtime_P95":float(np.percentile(lat,95)) if lat else None,
            "total_seconds":float(sum(r.get("seconds",0) for r in null+alt)),
            "errors":err,"identity_max":identity,"gates":gates,"certified":bool(all(gates.values()))}
    return report


def design():
    out={}
    for n in (3000,6000):
        cutoff=next(k-1 for k in range(n+1) if stats.binom.cdf(k,n,.07)>.05/24)
        out[str(n)]={"max_rejections":cutoff,"P_all24_size_gates_if_each_true_rate_.05_lower":max(0.,1-24*stats.binom.sf(cutoff,n,.05))}
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input",required=True);ap.add_argument("--phase",required=True);ap.add_argument("--out",required=True)
    a=ap.parse_args();rows=[];seen=set();hashes={}
    for f in sorted(Path(a.input).rglob("*.json")):
        if not f.name.startswith(("3PL_","4PL_")):continue
        raw=f.read_bytes();r=json.loads(raw);key=(r["model"],r["root"],r["r"])
        if key in seen:raise ValueError("duplicate independent dataset "+str(key))
        seen.add(key);rows.append(r);hashes[str(f.relative_to(a.input))]=hashlib.sha256(raw).hexdigest()
    report={"phase":a.phase,"source_run":"actual JSON records; dataset is MC cluster","models":{m:summarize(rows,m,a.phase) for m in ("3PL","4PL")},"precision_design":design(),"record_hashes":hashes}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(report,indent=1,default=lambda v:v.item()))
    print("P3_REPORT_JSON "+json.dumps({k:v for k,v in report.items() if k!="record_hashes"},default=lambda v:v.item()),flush=True)


if __name__=="__main__":main()
