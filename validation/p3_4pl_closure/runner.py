"""One independent native calibration per dataset, bounded/checkpointed."""
import argparse,hashlib,json,os,time,sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import mirt
from core.irt_python_mirt_adapter import fit_python_mirt
from validation.p3_empirical import constrained
from validation.p3_4pl_closure.dgp import data,KINDS
DEADLINE=1795705641  # 2026-11-26T15:07:21Z

def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v

def one(job):
    root,r,kind,noise,work=job;target=Path(work)/f"4PL_{kind}_{r:05d}.json"
    if target.exists():return json.loads(target.read_text())
    if time.time()>DEADLINE:raise TimeoutError("90-minute wall-clock deadline")
    x,seed=data(root,r,kind);start=time.perf_counter()
    rec=dict(model="4PL",kind=kind,root=root,r=r,seed=seed,source=os.getenv("GITHUB_SHA"),
        run=os.getenv("GITHUB_RUN_ID"),protocol_commit="e7f8b3b13501723cd638a07d8055d2cf0167b874",
        data_sha256=hashlib.sha256(x.tobytes()).hexdigest(),python=sys.version,mirt=mirt.__version__)
    try:
        f=fit_python_mirt(x,model="4PL",estimation="MAP",n_quadpts=61,max_iter=500,tol=1e-4).fit
        rec.update(converged=bool(f.converged),iterations=int(f.n_iterations))
        if not f.converged:raise ValueError("native nonconvergence")
        eta=f.coef()[["discrimination","difficulty","guessing","upper"]].to_numpy(float).ravel()
        native=mirt.itemfit(f,x,statistics=["S_X2"])
        np.savez_compressed(target.with_suffix(".npz"),eta=eta,counts=constrained.E.K.pattern_counts(x).astype(np.uint16))
        rec.update(eta=eta.tolist(),sx_native=native.S_X2.tolist(),p_chi=native.p_value.tolist(),df=native.df.tolist())
        result=constrained.analyse(x,eta,"4PL",[noise,4,r,KINDS.index(kind)])
        identity=float(np.max(abs(native.S_X2.to_numpy()-np.array([v["sx"] for v in result["items"]]))))
        if not np.isfinite(identity) or identity>1e-8:raise ValueError("native S-X2 identity")
        rec.update(result=result,identity=identity)
    except Exception as ex:rec["error"]=repr(ex)
    rec["seconds"]=time.perf_counter()-start
    tmp=target.with_suffix(".tmp");tmp.write_text(json.dumps(clean(rec)));tmp.replace(target)
    return rec

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--phase",choices=["pilot","confirm"],required=True)
    ap.add_argument("--shard",type=int,default=0);ap.add_argument("--shards",type=int,default=4);ap.add_argument("--work",required=True)
    z=ap.parse_args();Path(z.work).mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path("validation/p3_4pl_closure/PROTOCOL.json").read_text())[z.phase if z.phase=="pilot" else "confirmation"]
    R=cfg["null_R"];power=cfg["power_R_each"]
    jobs=[(cfg["null_root"] if kind=="null" else cfg["power_root"],r,kind,cfg["reference_root"],z.work)
          for kind in KINDS for r in range(z.shard,R if kind=="null" else power,z.shards)]
    with ProcessPoolExecutor(2) as pool:
        for rec in pool.map(one,jobs,chunksize=1):
            print(json.dumps({k:rec.get(k) for k in ("kind","r","converged","error","seconds")}),flush=True)
if __name__=="__main__":main()
