"""Shared native-fit development/baseline/confirmation; cached checkpoints, no duplicate fits."""
import argparse,hashlib,json,os,time,traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from validation.irt_closure.engine import base_fit,write_json,deadline
from validation.irt_closure.patterns import person_evidence
from validation.irt_closure.bootstrap import run_bootstrap

ROOT=Path(__file__).resolve().parent
PROTO=json.loads((ROOT/"PROTOCOL.json").read_text())


def worker(job):
    model,r,stage,work,do_boot,B,persons=job
    directory=Path(work);directory.mkdir(parents=True,exist_ok=True)
    target=directory/f"result_{r:05d}.json"
    signature=hashlib.sha256(b"".join((ROOT/n).read_bytes() for n in ("engine.py","patterns.py","bootstrap.py","runner.py"))).hexdigest()
    if target.exists():
        old=json.loads(target.read_text())
        if old.get("analysis_sha256")==signature:return old
    result={"model":model,"r":r,"stage":stage,"analysis_sha256":signature,
        "code_commit":os.getenv("GITHUB_SHA"),"run_id":os.getenv("GITHUB_RUN_ID"),
        "certification":"NOT CERTIFIED","p1":None,"p2":None,"p3":None}
    try:
        data,info=base_fit(model,r,stage,directory,persons)
        result["calibration"]=info
        if not info["converged"]:raise ValueError("Failed native calibration; retained")
        result["p3"]={"pvalue":data["pvalue"].tolist(),"sx":data["sx"].tolist(),"df":data["df"].tolist(),
                      "evaluable":np.isfinite(data["pvalue"]).tolist()}
        if persons:
            try:
                if "native_mean" not in data:raise ValueError(info.get("person_error","Native pattern scores unavailable"))
                result["p2"]=person_evidence(model,{q:data[q] for q in "abcd"},data["native_mean"],data["native_sd"])
            except Exception as e:result["p2_error"]=repr(e)
        if do_boot:
            try:
                result["p1"]=run_bootstrap(model,r,stage,data,directory,B)
            except Exception as e:
                result["p1_error"]=repr(e)
                result["p1_traceback"]=traceback.format_exc()
    except Exception as e:result["error"]=repr(e);result["traceback"]=traceback.format_exc()
    write_json(target,result)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--model",choices=["3PL","4PL"],required=True)
    ap.add_argument("--stage",choices=["pilot","baseline","confirm"],required=True)
    ap.add_argument("--shard",type=int,required=True);ap.add_argument("--shards",type=int,required=True)
    ap.add_argument("--total",type=int,required=True);ap.add_argument("--B",type=int,default=399)
    ap.add_argument("--work",required=True);ap.add_argument("--bootstrap",action="store_true")
    a=ap.parse_args();start=time.monotonic()
    ids=list(range(a.shard,a.total,a.shards))
    jobs=[(a.model,r,a.stage,a.work,a.bootstrap,a.B,a.stage=="pilot" or (a.stage=="baseline" and r<300)) for r in ids]
    results=[]
    try:
        with ProcessPoolExecutor(2) as pool:
            for result in pool.map(worker,jobs):
                results.append(result)
                print("PROGRESS",json.dumps({"r":result["r"],"model":a.model,"stage":a.stage,
                    "base":result.get("calibration",{}).get("converged",False),
                    "p1":bool(result.get("p1",{}).get("p1_valid")) if result.get("p1") else False,
                    "p2":result.get("p2") is not None,"error":result.get("error")}),flush=True)
    except Exception as e:print("EXECUTION_ERROR",repr(e),flush=True)
    out={"stage":a.stage,"model":a.model,"shard":a.shard,"expected":len(ids),"completed":len(results),
         "wall_seconds":time.monotonic()-start,"results":results}
    write_json(Path(a.work)/"summary.json",out)
    # Compact machine-readable result; raw native/bootstrap data are preserved as artifacts/cache.
    print("SUMMARY_JSON "+json.dumps({"stage":a.stage,"model":a.model,"shard":a.shard,
        "expected":len(ids),"completed":len(results),"wall_seconds":out["wall_seconds"],
        "base_valid":sum(r.get("calibration",{}).get("converged",False) for r in results),
        "p1_valid":sum(bool(r.get("p1",{}).get("p1_valid")) for r in results if r.get("p1")),
        "p2_valid":sum(r.get("p2") is not None for r in results)}),flush=True)
    return 0 if len(results)==len(ids) else 1


if __name__=="__main__":raise SystemExit(main())
