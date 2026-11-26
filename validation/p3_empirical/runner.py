"""One native fit per independent dataset, compressed checkpoints and explicit failures."""
import argparse,hashlib,importlib.metadata,json,os,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
os.environ.setdefault("OPENBLAS_NUM_THREADS","1");os.environ.setdefault("OMP_NUM_THREADS","1")
import numpy as np
import mirt
from validation.p3_sx2_bootstrap.run import data
from core.irt_python_mirt_adapter import fit_python_mirt
from validation.p3_empirical import kernel as E
COLS=["discrimination","difficulty","guessing","upper"]


def one(job):
    model,root,r,alt,work,expected=job;start=time.perf_counter()
    rec={"model":model,"root":root,"r":r,"alt":alt,"source":os.getenv("GITHUB_SHA"),"run":os.getenv("GITHUB_RUN_ID")}
    stem=Path(work)/f"{model}_{root}_{r:05d}"
    target=stem.with_suffix(".json")
    if target.exists():return json.loads(target.read_text())
    if time.time()>1795700849:raise TimeoutError("P3 hard stop 2026-11-26T13:47:29Z")
    x,seed=data(model,root,r,alt)
    rec.update(seed=seed,data_sha256=hashlib.sha256(x.tobytes()).hexdigest())
    try:
        assert importlib.metadata.version("mirt")=="1.2.0"
        f=fit_python_mirt(x,model=model,estimation="MAP",n_quadpts=61,max_iter=500,tol=1e-4).fit
        rec["converged"]=bool(f.converged);rec["iterations"]=int(f.n_iterations)
        if not f.converged:raise ValueError("native nonconvergence")
        eta=f.coef()[COLS[:4 if model=="4PL" else 3]].to_numpy(float).ravel()
        np.savez_compressed(stem.with_suffix(".npz"),counts=E.K.pattern_counts(x).astype(np.uint16),eta=eta,
            **({"responses":x} if root in (202611261001,202611261002) and r<10 else {}))
        native=mirt.itemfit(f,x,statistics=["S_X2"])
        res=E.analyse(x,eta,model)
        identity=float(np.max(abs(native.S_X2.to_numpy()-np.array([v["sx"] for v in res["items"]]))))
        if identity>1e-8:raise ValueError("native statistic identity "+str(identity))
        rec.update(result=res,identity=identity,sx_native=native.S_X2.tolist(),df=native.df.tolist(),p_chi=native.p_value.tolist())
        if expected:
            other=E.analyse(x,eta,model,expected=True)
            rec["expected_sensitivity"]={k:other[k] for k in ("diagnostics","seconds")}
            rec["p_expected"]=[v["p_emp"] for v in other["items"]]
    except Exception as ex:rec["error"]=repr(ex)
    rec["seconds"]=time.perf_counter()-start
    target.write_text(json.dumps(rec,allow_nan=True))
    return rec


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--model",required=True);ap.add_argument("--root",type=int,required=True)
    ap.add_argument("--R",type=int,required=True);ap.add_argument("--shard",type=int,default=0);ap.add_argument("--shards",type=int,default=1)
    ap.add_argument("--alt",action="store_true");ap.add_argument("--work",required=True);ap.add_argument("--workers",type=int,default=2)
    ap.add_argument("--expected",action="store_true");args=ap.parse_args();Path(args.work).mkdir(parents=True,exist_ok=True)
    jobs=[(args.model,args.root,r,args.alt,args.work,args.expected and r<5) for r in range(args.shard,args.R,args.shards)]
    with ProcessPoolExecutor(args.workers) as pool:
        for rec in pool.map(one,jobs,chunksize=1):
            print(json.dumps({k:rec.get(k) for k in ("model","r","converged","error","seconds")}),flush=True)


if __name__=="__main__":main()
