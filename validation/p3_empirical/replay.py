"""New candidates on frozen development calibrations; this script NEVER fits a model."""
import argparse,hashlib,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from validation.p3_sx2_bootstrap.run import data
from validation.p3_empirical import constrained,efficient


def one(job):
    file,out=job;old=json.loads(Path(file).read_text());target=Path(out)/Path(file).name
    if target.exists():return json.loads(target.read_text())
    rec={k:v for k,v in old.items() if k not in ("error","result","expected_sensitivity","p_expected")}
    rec["native_calibration_reused"]=True;rec["native_identity_available"]="identity" in old
    x,_=data(old["model"],old["root"],old["r"],old["alt"])
    assert hashlib.sha256(x.tobytes()).hexdigest()==old["data_sha256"]
    raw=Path(file).with_suffix(".npz")
    if not raw.exists():rec["error"]="Original calibration checkpoint missing";target.write_text(json.dumps(rec));return rec
    eta=np.load(raw)["eta"];start=time.perf_counter()
    rec["boundary_coordinates"]=[]
    k=4 if old["model"]=="4PL" else 3
    for i,value in enumerate(eta):
        p="abcd"[i%k];lo,hi=constrained.E.BOUNDS[p]
        if value-lo<=1e-6 or hi-value<=1e-6:rec["boundary_coordinates"].append({"item":i//k+1,"parameter":p,"value":float(value)})
    items=[{} for _ in range(12)];diag={}
    try:
        result=constrained.analyse(x,eta,old["model"],[202611261003,int(old["model"][0]),old["r"],int(old["alt"])])
        items=result["items"];diag["constrained"]=result["diagnostics"]
    except Exception as ex:diag["constrained_error"]=repr(ex)
    try:
        other=efficient.analyse(x,eta,old["model"])
        for j,item in enumerate(other["items"]):items[j].update(item)
    except Exception as ex:diag["efficient_error"]=repr(ex)
    rec.update(result={"items":items,"seconds":time.perf_counter()-start,"diagnostics":diag},seconds=time.perf_counter()-start,
        p_chi=old.get("p_chi",[None]*12),identity=old.get("identity",0.),
        method_names=["p_constrained","p_constrained_standardized","p_efficient"])
    target.write_text(json.dumps(rec));return rec


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input",required=True);ap.add_argument("--out",required=True)
    ap.add_argument("--model",required=True);ap.add_argument("--shard",type=int,default=0);ap.add_argument("--shards",type=int,default=2)
    z=ap.parse_args();Path(z.out).mkdir(parents=True,exist_ok=True)
    jobs=[]
    for f in sorted(Path(z.input).glob(z.model+"_*.json")):
        if json.loads(f.read_text())["r"]%z.shards==z.shard:jobs.append((str(f),z.out))
    with ProcessPoolExecutor(2) as pool:
        for r in pool.map(one,jobs):
            print(json.dumps({k:r.get(k) for k in ("model","r","alt","boundary_coordinates","error")}),flush=True)


if __name__=="__main__":main()
