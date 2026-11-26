"""Fixed final gates, all failures retained; dataset-cluster Monte Carlo uncertainty."""
import argparse,json,hashlib,gzip
from pathlib import Path
import numpy as np
from scipy import stats
from validation.p3_4pl_closure.dgp import KINDS
from validation.p3_sx2_aggregate.aggregate import cp_lower,cp_upper

def summarize(rows,method,phase):
    null=sorted([r for r in rows if r["kind"]=="null"],key=lambda r:r["r"])
    p=np.full((len(null),12),np.nan);lat=[];identity=0.
    for i,r in enumerate(null):
        if r.get("result") and not r.get("error"):
            p[i]=[x.get(method,np.nan) for x in r["result"]["items"]]
            p[i,~np.isfinite(np.array(r["p_chi"],float))]=np.nan
            lat.append(r["result"]["seconds"]);identity=max(identity,r["identity"])
    bad=~np.isfinite(p);dec=(p<.05)|bad;cluster=dec.mean(1);n=len(null)
    mcse=float(cluster.std(ddof=1)/np.sqrt(n)) if n>1 else 1.
    upp=[cp_upper(int(k),n,.05/24) for k in dec.sum(0)] if n else [1.]*12
    powers={}
    for kind in KINDS[1:]:
        rr=sorted([r for r in rows if r["kind"]==kind],key=lambda r:r["r"])
        pv=np.array([r.get("result",{}).get("items",[{}])[0].get(method,np.nan) if not r.get("error") else np.nan for r in rr],float)
        successes=int((pv<.05).sum());low=cp_lower(successes,len(rr),.025/3) if len(rr) else 0.
        powers[kind]=dict(R=len(rr),detected=successes,power=successes/max(1,len(rr)),MCSE=float(np.sqrt((successes/max(1,len(rr)))*(1-successes/max(1,len(rr)))/max(1,len(rr)))),
            lower=low,upper=cp_upper(successes,len(rr),.025/3) if len(rr) else 1.,refusal=float((~np.isfinite(pv)).mean()) if len(rr) else 1.,
            power_gate=bool(low>.10),complete=phase=="confirm" and len(rr)==600)
    gates=dict(size=max(upp)<=.07,MCSE=mcse<=.004,pooled_lower=float(cluster.mean())-stats.norm.ppf(.975)*mcse>=.03 if n else False,
        refusal=float(bad.mean())<=.01 if n else False,identity=identity<=1e-8,complete=phase=="confirm" and n==6000)
    return dict(method=method,phase=phase,null_R=n,TypeI=float(cluster.mean()) if n else None,MCSE=mcse,
        item_counts=dec.sum(0).tolist(),upper=upp,worst_upper=max(upp),refusal=float(bad.mean()) if n else None,
        dataset_refusal=float(bad.any(1).mean()) if n else None,converged=sum(bool(r.get("converged")) for r in rows),
        runtime_median=float(np.median(lat)) if lat else None,runtime_P95=float(np.percentile(lat,95)) if lat else None,
        identity_max=identity,power=powers,gates=gates,
        certified_claims=[k for k,v in powers.items() if all(gates.values()) and v["power_gate"] and v["complete"] and v["refusal"]<=.01],
        errors={e:sum(r.get("error")==e for r in rows) for e in set(r.get("error") for r in rows) if e})

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input",required=True);ap.add_argument("--phase",required=True);ap.add_argument("--out",required=True);z=ap.parse_args()
    rows=[];seen=set()
    for f in sorted(Path(z.input).rglob("4PL_*.json")):
        r=json.loads(f.read_text());key=(r["kind"],r["r"])
        if key in seen:raise ValueError("Duplicate dataset "+str(key))
        seen.add(key);rows.append(r)
    methods=["p_constrained","p_constrained_standardized"]
    if z.phase=="confirm":methods=[json.loads(Path("validation/p3_4pl_closure/SELECTION.json").read_text())["method"]]
    out=dict(phase=z.phase,records=len(rows),methods={m:summarize(rows,m,z.phase) for m in methods})
    target=Path(z.out);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(out,indent=1,default=lambda x:x.item()))
    print("P3N_REPORT_JSON "+json.dumps(out,default=lambda x:x.item()),flush=True)
if __name__=="__main__":main()
