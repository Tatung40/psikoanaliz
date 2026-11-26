"""Single prospectively selected 3PL confirmation; native fits checkpointed once."""
import argparse,hashlib,json,os,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import mirt
from validation.p3_sx2_bootstrap.run import data
from core.irt_python_mirt_adapter import fit_python_mirt
from validation.p3_empirical import constrained


def one(job):
    root,r,alt,work=job;model="3PL";target=Path(work)/f"{model}_{root}_{r:05d}.json"
    if target.exists():return json.loads(target.read_text())
    if time.time()>1795700849:raise TimeoutError("Hard wall-clock stop")
    x,seed=data(model,root,r,alt);start=time.perf_counter()
    rec={"model":model,"root":root,"r":r,"alt":alt,"seed":seed,"source":os.getenv("GITHUB_SHA"),"run":os.getenv("GITHUB_RUN_ID"),
        "protocol_commit":"5be1faad9b40d27373b34ff7c887030bf64c89ca","data_sha256":hashlib.sha256(x.tobytes()).hexdigest()}
    try:
        f=fit_python_mirt(x,model=model,estimation="MAP",n_quadpts=61,max_iter=500,tol=1e-4).fit
        rec.update(converged=bool(f.converged),iterations=int(f.n_iterations),mirt=mirt.__version__)
        if not f.converged:raise ValueError("native nonconvergence")
        eta=f.coef()[["discrimination","difficulty","guessing"]].to_numpy(float).ravel()
        tab=mirt.itemfit(f,x,statistics=["S_X2"])
        rec.update(sx_native=tab.S_X2.tolist(),df=tab.df.tolist(),p_chi=tab.p_value.tolist())
        np.savez_compressed(target.with_suffix(".npz"),eta=eta,counts=constrained.E.K.pattern_counts(x).astype(np.uint16),sx=tab.S_X2.to_numpy(),df=tab.df.to_numpy())
        result=constrained.analyse(x,eta,model,[202611262003,3,r,int(alt)])
        identity=float(np.max(abs(np.array([i["sx"] for i in result["items"]])-tab.S_X2.to_numpy())))
        if not np.isfinite(identity) or identity>1e-8:raise ValueError("native statistic identity "+str(identity))
        rec.update(result=result,identity=identity,native_identity_available=True)
    except Exception as ex:rec["error"]=repr(ex)
    rec["seconds"]=time.perf_counter()-start
    target.write_text(json.dumps(rec));return rec


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=int,required=True);ap.add_argument("--R",type=int,required=True)
    ap.add_argument("--shard",type=int,required=True);ap.add_argument("--alt",action="store_true");ap.add_argument("--work",required=True)
    z=ap.parse_args();Path(z.work).mkdir(parents=True,exist_ok=True)
    jobs=[(z.root,r,z.alt,z.work) for r in range(z.shard,z.R,20)]
    with ProcessPoolExecutor(2) as pool:
        for r in pool.map(one,jobs,chunksize=1):
            print(json.dumps({k:r.get(k) for k in ("r","alt","converged","error","seconds")}),flush=True)


if __name__=="__main__":main()
