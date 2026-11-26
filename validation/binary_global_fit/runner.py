"""One native calibration per new global-fit dataset; failed attempts are retained."""
import argparse,json,hashlib,os,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import mirt
from mirt.diagnostics.modelfit import compute_fit_indices
from core.irt_python_mirt_adapter import fit_python_mirt
from validation.binary_global_fit.dgp import data,KINDS
from validation.binary_global_fit.analysis import analyse
DEADLINE=1795713798 # 2026-11-26T17:23:18Z
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def one(job):
    model,root,r,kind,noise,work=job;path=Path(work)/f"{model}_{kind}_{r:05d}.json"
    if path.exists():return json.loads(path.read_text())
    if time.time()>DEADLINE:raise TimeoutError("Hard120-minute stop")
    x,seed=data(model,root,r,kind);start=time.perf_counter()
    rec=dict(model=model,kind=kind,r=r,root=root,seed=seed,source=os.getenv("GITHUB_SHA"),run=os.getenv("GITHUB_RUN_ID"),
             data_sha256=hashlib.sha256(x.tobytes()).hexdigest(),protocol_commit="521be0af635198e7bc0ab326b47e0ab7a066ca86",mirt=mirt.__version__)
    try:
        f=fit_python_mirt(x,model=model,estimation="MAP",n_quadpts=61,max_iter=500,tol=1e-4).fit
        rec.update(converged=bool(f.converged),iterations=int(f.n_iterations))
        if not f.converged:raise ValueError("native nonconvergence")
        eta=f.coef()[["discrimination","difficulty","guessing","upper"][:3 if model=="3PL" else 4]].to_numpy(float).ravel()
        idx=compute_fit_indices(f,x,n_quadpts=61)
        rec.update(eta=eta.tolist(),native=idx)
        np.savez_compressed(path.with_suffix(".npz"),eta=eta,counts=np.bincount(x@(1<<np.arange(12)),minlength=4096).astype(np.uint16))
        result=analyse(x,eta,model,[noise,int(model[0]),r,KINDS.index(kind)])
        error=abs(result["M2"]-idx["M2"])
        if result["df"]!=idx["M2_df"] or error>1e-7+1e-6*abs(idx["M2"]):raise ValueError("Native M2 mathematical identity/rank mismatch")
        rec.update(result=result,identity=error)
    except Exception as e:rec["error"]=repr(e)
    rec["seconds"]=time.perf_counter()-start
    path.write_text(json.dumps(clean(rec)));return rec
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--phase",choices=["pilot","confirm"],required=True);ap.add_argument("--shard",type=int,required=True)
    ap.add_argument("--shards",type=int,required=True);ap.add_argument("--work",required=True);z=ap.parse_args()
    cfg=json.loads(Path("validation/binary_global_fit/PROTOCOL.json").read_text())[z.phase if z.phase=="pilot" else "confirmation"]
    Path(z.work).mkdir(parents=True,exist_ok=True)
    jobs=[(model,cfg["null_root"] if kind=="null" else cfg["power_root"],r,kind,cfg["noise_root"],z.work)
          for model in ("3PL","4PL") for kind in KINDS for r in range(z.shard,cfg["null_R"] if kind=="null" else cfg["power_R_each"],z.shards)]
    with ProcessPoolExecutor(2) as pool:
        for r in pool.map(one,jobs,chunksize=1):print(json.dumps({k:r.get(k) for k in ("model","kind","r","converged","error","seconds")}),flush=True)
if __name__=="__main__":main()
