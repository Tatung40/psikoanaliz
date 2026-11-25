"""Native calibration/cache only; independent stage seeds and no silent refits."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import mirt
from core.irt_python_mirt_adapter import fit_python_mirt
from core.irt_binary_persons import binary_person_abilities
from validation.binary_certification.common import truth,simulate
from validation.irt_closure.patterns import PATTERNS
COLS={"a":"discrimination","b":"difficulty","c":"guessing","d":"upper"}
ROOTS={"pilot":202611251001,"baseline":202611251101,"confirm":202611251201}
DEADLINE=1795620122  # 2026-11-25T15:22:02Z


def deadline():
    if time.time()>DEADLINE-15:raise TimeoutError("Preregistered global deadline")


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def clean(v):
        if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [clean(x) for x in v]
        if isinstance(v,np.ndarray):return clean(v.tolist())
        if isinstance(v,(np.integer,)):return int(v)
        if isinstance(v,(np.floating,float)):return float(v) if np.isfinite(v) else None
        return v
    tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(clean(obj),ensure_ascii=False,indent=1));tmp.replace(path)


def native_statistics(f,x):
    try:
        table=mirt.itemfit(f,x,statistics=["S_X2"])
        if not {"S_X2","df","p_value"}<=set(table.columns):
            raise ValueError("Actual native S-X2 API columns: "+str(table.columns))
        return table["S_X2"].to_numpy(float),table["p_value"].to_numpy(float),table["df"].to_numpy(float),None
    except Exception as e:return np.full(x.shape[1],np.nan),np.full(x.shape[1],np.nan),np.full(x.shape[1],np.nan),repr(e)


def fit_once(x,model,start=None):
    deadline()
    # Cold/default is canonical; warm/squarem permitted only after separate benchmark.
    kw={} if start is None else {"start_values":start}
    return fit_python_mirt(x,model=model,estimation="MAP",n_quadpts=61,max_iter=500,tol=1e-4,**kw)


def coefficients(f,model):
    co=f.coef();k=len(co)
    return {q:co[col].to_numpy(float) if col in co else np.ones(k) for q,col in COLS.items()}


def canonical_covariance(f,p,model):
    qs="abc" if model=="3PL" else "abcd"
    co=f.coef();items=list(map(str,co.index))
    labels=list(f.vcov_labels() if callable(f.vcov_labels) else f.vcov_labels)
    reverse={v:k for k,v in COLS.items()}
    parsed=[]
    for label in labels:
        name,_,item=label.partition("[");parsed.append((reverse[name],items.index(item.rstrip("]"))))
    order=[parsed.index((q,j)) for j in range(12) for q in qs]
    V=np.asarray(f.vcov() if callable(f.vcov) else f.vcov,float)
    return V[np.ix_(order,order)]


def base_fit(model,r,stage,work,persons=False):
    path=Path(work)/f"base_{r:05d}.npz";meta=path.with_suffix(".json")
    if path.exists() and meta.exists():
        return dict(np.load(path,allow_pickle=False)),json.loads(meta.read_text())
    deadline();seed=[ROOTS[stage],int(model[0]),r,0]
    x=simulate(model,3000,12,seed).astype(np.uint8)
    data={"x":x};info={"r":r,"seed":seed,"converged":False,"fit_called":False,
        "source_commit":os.getenv("GITHUB_SHA"),"run_id":os.getenv("GITHUB_RUN_ID"),"data_sha256":digest(x)}
    start=time.monotonic()
    try:
        if mirt.__version__!="1.2.0":raise ValueError("Unverified native version")
        info["fit_called"]=True
        outcome=fit_once(x,model);f=outcome.fit
        info["converged"]=bool(f.converged)
        info["iterations"]=int(f.n_iterations)
        if not f.converged:raise ValueError("Native calibration did not converge")
        p=coefficients(f,model);data.update(p)
        data["V"]=canonical_covariance(f,p,model)
        sx,pv,df,error=native_statistics(f,x)
        data.update(sx=sx,pvalue=pv,df=df)
        info["sx_error"]=error
        if persons:
            try:
                people=binary_person_abilities(outcome,PATTERNS,[f"P{i:04d}" for i in range(4096)])
                data["native_mean"]=people["Yetenek (θ)"].to_numpy(float)
                data["native_sd"]=people["Posterior SS (θ)"].to_numpy(float)
                info["person_numerical_verified"]=True
            except Exception as e:info["person_error"]=repr(e)
        info["log_likelihood"]=float(f.log_likelihood)
        info["mirt"]=mirt.__version__
    except Exception as e:info["error"]=repr(e)
    info["seconds"]=time.monotonic()-start
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**data);write_json(meta,info)
    return data,info
