"""H: exact native-box coordinates, uncentered compressed marginal-score sandwich."""
import argparse,hashlib,importlib.util,importlib.metadata,json,os,sys,time,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.special import expit,logit
from core.irt_binary_inference import Z95
from validation.binary_certification.common import truth,simulate
from validation.binary_certification.reference_numpy import PRIOR,BOX,objective
SPEC=importlib.util.spec_from_file_location("frozen_g",os.environ.get("G_REFERENCE","/tmp/parameter-g-runner.py"))
G=importlib.util.module_from_spec(SPEC);sys.modules["frozen_g"]=G;SPEC.loader.exec_module(G)
PROTOCOL_COMMIT="b1a40c916e2f99b35ef5832a5ba4175691829c96"


def coordinates(v,bounds):
    bounds=np.asarray(bounds,float);lo,hi=bounds.T
    u=(v-lo)/(hi-lo)
    if np.any((u<=0)|(u>=1)):raise ValueError("Boundary coordinate")
    eta=logit(u);D=(hi-lo)*u*(1-u);h=D*(1-2*u)
    return eta,D,h


def transformed_covariance(V,U,counts,D,h,g):
    A=np.linalg.inv(V)
    Aeta=D[:,None]*A*D[None,:]-np.diag(g*h)
    scores=U*D
    B=scores.T@(counts[:,None]*scores)  # uncentered, exact frequency weights
    for label,M in (("bread",Aeta),("meat",B)):
        ev=np.linalg.eigvalsh((M+M.T)/2)
        if not np.isfinite(M).all() or ev[0]<=0 or ev[-1]/ev[0]>1e8:
            raise ValueError(label+" PD/condition gate")
    inverse=np.linalg.inv(Aeta)
    C=inverse@B@inverse.T;C=(C+C.T)/2
    ev=np.linalg.eigvalsh(C)
    if not np.isfinite(C).all() or ev[0]<=0 or ev[-1]/ev[0]>1e8:
        raise ValueError("Transformed sampling covariance gate")
    return C,Aeta,B


def numerical_bread_audit(x,model,p,parsed,A):
    f,qs,k=objective(x,model,points=61)
    order=[(q,j) for j in range(k) for q in qs]
    perm=[parsed.index(pair) for pair in order]
    v=np.array([p[q][j] for q,j in order])
    limits=np.array([BOX[q] for q,j in order])
    distance=np.minimum(v-limits[:,0],limits[:,1]-v)
    hs=np.minimum(1e-4*np.maximum(1,abs(v)),.1*distance)
    matrices=[]
    for multiplier in (1.,2.):
        H=np.empty((len(v),len(v)))
        for i,h in enumerate(hs*multiplier):
            e=np.zeros(len(v));e[i]=h
            H[:,i]=(f(v+e)[1]-f(v-e)[1])/(2*h)
        matrices.append((H+H.T)/2)
    ref=A[np.ix_(perm,perm)]
    relative=[float(np.linalg.norm(H-ref)/np.linalg.norm(H)) for H in matrices]
    step_difference=float(np.linalg.norm(matrices[0]-matrices[1])/np.linalg.norm(matrices[0]))
    return {"native_relative_errors":relative,"two_step_relative_difference":step_difference,
            "pass":max(relative)<=1e-3 and step_difference<=1e-3}


def replicate(job):
    model,n,k,r,directory=job
    from core.irt_python_mirt_adapter import fit_python_mirt
    from mirt.estimation.base import _ITEM_PARAMETER_BOUNDS as bounds
    seed=[202611250801 if model=="3PL" else 202611250802,n,k,r]
    start=time.monotonic();record={"r":r,"seed":seed,"valid":False,"converged":False,"audit_seconds":0.}
    try:
        if importlib.metadata.version("mirt")!="1.2.0":raise ValueError("Unverified engine version")
        for col,q in G.SHORT.items():
            if tuple(bounds[col])!=BOX[q]:raise ValueError("Native bound mismatch")
        if not (0<BOX["c"][0]<BOX["c"][1]<=BOX["d"][0]<BOX["d"][1]<1):
            raise ValueError("Native boxes do not enforce c<d")
        x=simulate(model,n,k,seed)
        record["data_sha256"]=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
        f=fit_python_mirt(x,model=model,estimation="MAP",n_quadpts=61,max_iter=500,tol=1e-4).fit
        record["converged"]=bool(f.converged)
        if not f.converged:raise ValueError("nonconvergence")
        co=f.coef();items=list(map(str,co.index))
        p={q:co[col].to_numpy(float) if col in co else np.ones(k) for col,q in G.SHORT.items()}
        labels=list(f.vcov_labels() if callable(f.vcov_labels) else f.vcov_labels)
        V=np.asarray(f.vcov() if callable(f.vcov) else f.vcov,float)
        parsed=[];limits=[]
        for label in labels:
            kind,_,name=label.partition("[");j=items.index(name.rstrip("]"));q=G.SHORT[kind]
            lo,hi=bounds[kind];value=p[q][j]
            if min(abs(value-lo),abs(value-hi))<=1e-6:raise ValueError("Active boundary")
            parsed.append((q,j));limits.append((lo,hi))
        ev=np.linalg.eigvalsh((V+V.T)/2)
        if not np.isfinite(V).all() or ev[0]<=0 or ev[-1]/ev[0]>1e8:raise ValueError("Native covariance gate")
        patterns,counts=np.unique(x,axis=0,return_counts=True)
        marginal=G.marginal_scores(patterns,p,points=61)
        U=np.column_stack([marginal[q][:,j] for q,j in parsed])
        values=np.array([p[q][j] for q,j in parsed])
        eta,D,h=coordinates(values,limits)
        g=counts@U+np.array([PRIOR[q][1](p[q][j]) for q,j in parsed])
        C,Aeta,B=transformed_covariance(V,U,counts,D,h,g)
        record["patterns"]=len(patterns)
        record["max_posterior_score"]=float(abs(g).max())
        if r==0:
            audit_start=time.monotonic()
            record["native_bread_audit"]=numerical_bread_audit(x,model,p,parsed,np.linalg.inv(V))
            record["audit_seconds"]=time.monotonic()-audit_start
            if not record["native_bread_audit"]["pass"]:raise ValueError("Native bread numerical audit")
        limits=np.asarray(limits);width=limits[:,1]-limits[:,0]
        low=limits[:,0]+width*expit(eta-Z95*np.sqrt(np.diag(C)))
        high=limits[:,0]+width*expit(eta+Z95*np.sqrt(np.diag(C)))
        se=np.sqrt(np.diag(D[:,None]*C*D[None,:]))
        if np.any(low<=limits[:,0]) or np.any(high>=limits[:,1]):raise ValueError("Degenerate transformed endpoint")
        true=truth(model,k);record["by_parameter"]={}
        for q in ("a","b","c","d"):
            pos=[i for i,(name,j) in enumerate(parsed) if name==q]
            if not pos:continue
            js=[parsed[i][1] for i in pos];target=true[q][js]
            record["by_parameter"][q]={"items":js,"estimate":p[q][js].tolist(),
                "error":(p[q][js]-target).tolist(),"se":se[pos].tolist(),
                "covered":((low[pos]<=target)&(target<=high[pos])).astype(int).tolist()}
        record["valid"]=True
        np.savez_compressed(Path(directory)/f"joint_{r:04d}.npz",labels=np.array(labels),V=V,Aeta=Aeta,
                            B=B,Ceta=C,limits=limits,eta=eta,gradient=g,**p)
    except Exception as exc:
        record["error"]=repr(exc);record["traceback"]=traceback.format_exc()
    record["seconds"]=time.monotonic()-start
    return record


def run(model,n,directory):
    output=Path(directory);output.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();records=[];reason="complete";planned=300;k=12
    with ProcessPoolExecutor(2) as pool,(output/"replications.jsonl").open("w") as stream:
        while len(records)<planned:
            count=10 if not records else min(4,planned-len(records));begin=len(records)
            for record in pool.map(replicate,[(model,n,k,r,str(output)) for r in range(begin,begin+count)]):
                records.append(record);stream.write(json.dumps(record,allow_nan=False)+"\n");stream.flush()
            elapsed=time.monotonic()-start
            if len(records)==10:
                overhead=sum(r["audit_seconds"] for r in records)
                projected=overhead+sum(r["seconds"]-r["audit_seconds"] for r in records)/10*planned/2
                if projected>300:reason="preregistered feasibility budget";break
            if elapsed>=300:reason="preregistered execution budget";break
    report=G.summarize(records,model,n,k,planned,reason)
    report["protocol_commit"]=PROTOCOL_COMMIT
    report["native_bread_audit"]=records[0].get("native_bread_audit",{"pass":False})
    report["checks"]["native_bread_audit"]=report["native_bread_audit"]["pass"]
    report["cell_pass"]=all(report["checks"].values())
    report["status"]="SCOPED CELL CRITERIA PASS / NOT CERTIFIED" if report["cell_pass"] else "NOT CERTIFIED"
    report["wall_seconds"]=time.monotonic()-start
    report["versions"]={q:importlib.metadata.version(q) for q in ("mirt","numpy","scipy")}
    report["method"]="uncentered compressed empirical sandwich, exact native-box-logit marginal intervals"
    report["api"]="fit.vcov / fit.vcov_labels (callable or property), physical parameterization"
    report["protocol_sha256"]=hashlib.sha256(Path(__file__).with_name("PROTOCOL.json").read_bytes()).hexdigest()
    report["P2"]="NOT EXECUTED";report["P3"]="NOT EXECUTED"
    (output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("RESULT_JSON "+json.dumps(report,allow_nan=False))


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--model",choices=["3PL","4PL"],required=True)
    ap.add_argument("--n",type=int,choices=[3000,10000],required=True);ap.add_argument("--output",required=True)
    a=ap.parse_args();run(a.model,a.n,a.output)
