"""G: centered marginal-score covariance; validation only, never production inference."""
import argparse, hashlib, importlib.metadata, json, os, time, traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.special import expit,logsumexp,roots_hermite
from core.irt_binary_inference import Z95
from validation.binary_certification.common import simulate,truth
from validation.coverage_precision import coverage_precision
PROTOCOL_COMMIT="7c6dba3ccfa9475260053938e997f0c5d57e6078"
SHORT={"discrimination":"a","difficulty":"b","guessing":"c","upper":"d"}


def marginal_scores(x,p,points=61):
    x=np.asarray(x,float)
    z,w=roots_hermite(points);theta=np.sqrt(2)*z
    logistic=expit(p["a"]*(theta[:,None]-p["b"]))
    prob=np.clip(p["c"]+(p["d"]-p["c"])*logistic,1e-14,1-1e-14)
    ll=x@np.log(prob).T+(1-x)@np.log1p(-prob).T+np.log(w/np.sqrt(np.pi))
    posterior=np.exp(ll-logsumexp(ll,axis=1,keepdims=True))
    common=(p["d"]-p["c"])*logistic*(1-logistic)
    dp={"a":common*(theta[:,None]-p["b"]),"b":-common*p["a"],
        "c":1-logistic,"d":logistic}
    scores={q:x*(posterior@(v/(prob*(1-prob))))-posterior@(v/(1-prob)) for q,v in dp.items()}
    return scores


def empirical_sandwich(V,scores):
    V=np.asarray(V,float);scores=np.asarray(scores,float)
    if len(scores)<2 or V.shape!=(scores.shape[1],scores.shape[1]):
        raise ValueError("Invalid joint covariance/score layout")
    centered=scores-scores.mean(0)
    meat=(len(scores)/(len(scores)-1))*(centered.T@centered)
    S=V@meat@V.T
    S=(S+S.T)/2
    for label,matrix in (("posterior",V),("sampling",S)):
        if not np.isfinite(matrix).all():raise ValueError(label+" nonfinite")
        ev=np.linalg.eigvalsh((matrix+matrix.T)/2)
        if ev[0]<=0 or ev[-1]/ev[0]>1e8:raise ValueError(label+" covariance gate")
    return S,meat


def replicate(job):
    model,n,k,r,directory=job
    from core.irt_python_mirt_adapter import fit_python_mirt
    from mirt.estimation.base import _ITEM_PARAMETER_BOUNDS as bounds
    seed=[202611250701 if model=="3PL" else 202611250702,n,k,r]
    start=time.monotonic();record={"r":r,"seed":seed,"valid":False,"converged":False}
    try:
        x=simulate(model,n,k,seed)
        record["data_sha256"]=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
        outcome=fit_python_mirt(x,model=model,estimation="MAP",n_quadpts=61,tol=1e-4,max_iter=500)
        f=outcome.fit;record["converged"]=bool(f.converged)
        if not f.converged:raise ValueError("nonconvergence")
        co=f.coef();items=list(map(str,co.index))
        p={q:co[col].to_numpy(float) if col in co else np.ones(k) for col,q in SHORT.items()}
        labels=list(f.vcov_labels() if callable(f.vcov_labels) else f.vcov_labels)
        V=np.asarray(f.vcov() if callable(f.vcov) else f.vcov,float)
        marginal=marginal_scores(x,p)
        parsed=[]
        for label in labels:
            kind,_,name=label.partition("[");j=items.index(name.rstrip("]"))
            if kind not in SHORT:raise ValueError("Unrecognized physical covariance label")
            q=SHORT[kind];value=p[q][j];lo,hi=bounds[kind]
            if min(abs(value-lo),abs(value-hi))<=1e-6:raise ValueError("active bound")
            parsed.append((q,j))
        U=np.column_stack([marginal[q][:,j] for q,j in parsed])
        S,meat=empirical_sandwich(V,U)
        se=np.sqrt(np.diag(S));true=truth(model,k)
        record["by_parameter"]={}
        for q in ("a","b","c","d"):
            positions=[i for i,(name,j) in enumerate(parsed) if name==q]
            if not positions:continue
            js=[parsed[i][1] for i in positions]
            est=p[q][js];target=true[q][js];sd=se[positions]
            record["by_parameter"][q]={"items":js,"estimate":est.tolist(),
                "error":(est-target).tolist(),"se":sd.tolist(),
                "covered":(abs(est-target)<=Z95*sd).astype(int).tolist()}
        record["valid"]=True
        record["max_mean_likelihood_score"]=float(abs(U.mean(0)).max())
        np.savez_compressed(Path(directory)/f"joint_{r:04d}.npz",labels=np.array(labels),
                            V=V,meat=meat,S=S,**p)
    except Exception as exc:
        record["error"]=repr(exc)
        record["traceback"]=traceback.format_exc()
    record["seconds"]=time.monotonic()-start
    return record


def summarize(records,model,n,k,planned,reason):
    m=len(records);valid=[r for r in records if r["valid"]]
    out={"model":model,"N":n,"J":k,"planned":planned,"attempted":m,
         "valid":len(valid),"converged":sum(r["converged"] for r in records),
         "se_refused":m-len(valid),"stop_reason":reason,"parameter":{},
         "protocol_commit":PROTOCOL_COMMIT,"execution_commit":os.getenv("GITHUB_SHA"),
         "run_id":os.getenv("GITHUB_RUN_ID"),"certification":"NOT CERTIFIED"}
    checks={"complete":m==planned,"convergence":m>0 and out["converged"]/m>=.99,
            "refusal":m>0 and out["se_refused"]/m<=.01}
    for q in ("a","b","c","d"):
        parts=[np.column_stack([r["by_parameter"][q]["error"],
                 np.zeros(len(r["by_parameter"][q]["error"])),
                 r["by_parameter"][q]["covered"]]) for r in valid if q in r["by_parameter"]]
        if not parts:continue
        arr=np.vstack(parts);coverage=float(arr[:,-1].mean());bias=float(arr[:,0].mean())
        rmse=float(np.sqrt(np.mean(arr[:,0]**2)))
        precision=coverage_precision(parts,m,eligible_column=1)
        estimates=np.asarray([r["by_parameter"][q]["estimate"] for r in valid])
        ses=np.asarray([r["by_parameter"][q]["se"] for r in valid])
        sampling_var=np.var(estimates,axis=0,ddof=1) if len(valid)>1 else np.full(estimates.shape[1],np.nan)
        stats={"coverage":coverage,"bias":bias,"rmse":rmse,
            "failed_as_noncoverage":float(arr[:,-1].sum()/(m*k)),
            "withheld_rate":(m-len(valid))/m,**precision,
            "mean_se":float(ses.mean()),
            "empirical_sd":float(np.sqrt(sampling_var.mean())) if len(valid)>1 else None,
            "se_to_empirical_sd":float(np.sqrt(np.mean(ses**2)/sampling_var.mean())) if len(valid)>1 else None,
            "per_item_coverage":np.mean([r["by_parameter"][q]["covered"] for r in valid],axis=0).tolist()}
        out["parameter"][q]=stats
        checks[q+"_coverage"]=abs(coverage-.95)<=.02
        checks[q+"_precision"]=precision["coverage_mc_se"] is not None and precision["coverage_mc_se"]<=.005
        checks[q+"_withheld"]=stats["withheld_rate"]<=.05
        checks[q+"_rmse"]=rmse <= (.30 if q in ("a","b") else .07)
        if q in ("a","b"):checks[q+"_bias"]=abs(bias)<=.05
    checks["all_parameters"]=len(out["parameter"])==(3 if model=="3PL" else 4)
    out["checks"]=checks
    out["cell_pass"]=all(checks.values())
    out["status"]="CELL CRITERIA PASS / MODEL NOT CERTIFIED" if out["cell_pass"] else (
        "INCONCLUSIVE" if m!=planned or any(not v for key,v in checks.items() if key.endswith("_precision")) else "FAILED CRITERIA")
    return out


def run(model,n,k,directory):
    output=Path(directory);output.mkdir(parents=True,exist_ok=True)
    planned=300 if n==3000 else 150
    start=time.monotonic();records=[];reason="complete"
    with ProcessPoolExecutor(2) as pool, (output/"replications.jsonl").open("w") as stream:
        while len(records)<planned:
            count=10 if not records else min(4,planned-len(records))
            begin=len(records)
            batch=pool.map(replicate,[(model,n,k,r,str(output)) for r in range(begin,begin+count)])
            for record in batch:
                records.append(record);stream.write(json.dumps(record,allow_nan=False)+"\n");stream.flush()
            elapsed=time.monotonic()-start
            if len(records)==10 and elapsed/10*planned>240:
                reason="preregistered feasibility budget";break
            if elapsed>=240:
                reason="preregistered execution budget";break
    report=summarize(records,model,n,k,planned,reason)
    report["wall_seconds"]=time.monotonic()-start
    report["versions"]={q:importlib.metadata.version(q) for q in ("mirt","numpy","scipy")}
    report["protocol_sha256"]=hashlib.sha256(Path(__file__).with_name("PROTOCOL.json").read_bytes()).hexdigest()
    (output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("RESULT_JSON "+json.dumps(report,allow_nan=False))
    # Scientific failure is retained in JSON; a completed honest study is not a certification claim.
    return 0


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--model",choices=["3PL","4PL"],required=True)
    ap.add_argument("--n",type=int,choices=[3000,10000],required=True)
    ap.add_argument("--j",type=int,choices=[12,30],required=True);ap.add_argument("--output",required=True)
    a=ap.parse_args();raise SystemExit(run(a.model,a.n,a.j,a.output))
