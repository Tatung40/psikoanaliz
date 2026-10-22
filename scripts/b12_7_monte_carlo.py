"""B12.7 reproducible mixed-IRT Monte Carlo.

CLI: python -m scripts.b12_7_monte_carlo --truth data/b12_7_truth.csv --n 500 1000 2500 5000 --reps 100 --out mc_b12_7
Simulation and calibration must use the same item metric, while the M14/M15 testlet is a separate misspecification condition.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from core.irt_mixed import fit_mixed

ITEMS=[f"M{i:02d}" for i in range(1,31)]
TYPES=["1PL"]*8+["2PL"]*8+["3PL"]*8+["4PL"]*6
DIF=("M05","M12","M20")
PARAMS=("a","b","c","d")

def load_truth(path):
    t=pd.read_csv(path)
    required={"Madde","grup",*PARAMS}
    if not required.issubset(t.columns):raise ValueError("Truth requires Madde, grup and a,b,c,d.")
    if t.duplicated(["Madde","grup"]).any():raise ValueError("Duplicate truth entries.")
    for group in ("Kadin","Erkek"):
        s=t[t.grup==group]
        if set(s.Madde)!=set(ITEMS):raise ValueError(f"Missing/extra truth item for {group}.")
        if not np.isfinite(s[list(PARAMS)].to_numpy(dtype=float)).all():raise ValueError("Nonfinite truth.")
    return t

def generate(n,truth,seed,local_dependence=False):
    if n<100 or n%2:raise ValueError("N must be even and >=100.")
    rng=np.random.default_rng(seed);out=[]
    for group,mu,sd in (("Kadin",0.,1.),("Erkek",-.3,1.1)):
        tab=truth[truth.grup==group].set_index("Madde").loc[ITEMS]
        a,b,c,d=(tab[p].to_numpy(float) for p in PARAMS)
        theta=rng.normal(mu,sd,n//2)
        eta=a*(theta[:,None]-b)
        if local_dependence:
            eta[:,[13,14]]+=rng.normal(0,1.5,n//2)[:,None]
        probabilities=c+(d-c)*expit(eta)
        ans=(rng.random(probabilities.shape)<probabilities).astype(int)
        out.append(pd.DataFrame(ans,columns=ITEMS).assign(Cinsiyet=group))
    return pd.concat(out,ignore_index=True)

def run_one(n,rep,condition,truth,master_seed,starts=1):
    seed=int(np.random.SeedSequence([master_seed,n,rep,int(condition=="testlet")]).generate_state(1)[0])
    d=generate(n,truth,seed,local_dependence=condition=="testlet")
    if (d.groupby("Cinsiyet")[ITEMS].nunique()<2).any().any():
        return {"N":n,"rep":rep,"condition":condition,"seed":seed,"status":"constant_item","converged":False}
    t0=time.perf_counter()
    try:
        f=fit_mixed(d[ITEMS],TYPES,groups=d.Cinsiyet,reference="Kadin",focal="Erkek",
            group_specific=DIF,quadrature="mirt",starts=starts,seed=seed)
        row={"N":n,"rep":rep,"condition":condition,"seed":seed,"status":"ok",
             "converged":bool(f["converged"]),"kkt_per_person":f["kkt_per_person"],
             "loglik":f["loglik"],"boundary_count":len(f["bound"]),
             "elapsed_seconds":time.perf_counter()-t0}
        for group,tab in (("Kadin",f["items"]),("Erkek",f["items_focal"])):
            actual=truth[truth.grup==group].set_index("Madde").loc[ITEMS]
            est=tab.set_index("Madde").loc[ITEMS]
            for typ in ("1PL","2PL","3PL","4PL"):
                sub=np.array(TYPES)==typ
                for p in {"1PL":("b",),"2PL":("a","b"),"3PL":("a","b","c"),"4PL":PARAMS}[typ]:
                    err=est[p].to_numpy()[sub]-actual[p].to_numpy()[sub]
                    row[f"{group}_{typ}_{p}_bias"]=float(err.mean())
                    row[f"{group}_{typ}_{p}_rmse"]=float(np.sqrt(np.mean(err**2)))
        row["focal_mean_error"]=f["latent_focal"][0]+.3
        row["focal_sd_error"]=f["latent_focal"][1]-1.1
        return row
    except Exception as e:
        return {"N":n,"rep":rep,"condition":condition,"seed":seed,"status":"error",
                "error":f"{type(e).__name__}: {str(e)[:250]}","converged":False,
                "elapsed_seconds":time.perf_counter()-t0}

def summarize(df):
    if df.empty:return pd.DataFrame()
    rows=[]
    for (n,cond),g in df.groupby(["N","condition"]):
        good=g[(g.status=="ok") & g.converged.fillna(False).astype(bool)]
        r={"N":n,"condition":cond,"attempted":len(g),"successful":len(good),
           "convergence_rate":len(good)/len(g),"failure_count":len(g)-len(good)}
        for p in ("boundary_count","elapsed_seconds"):
            if len(good):r["mean_"+p]=float(good[p].mean())
        for col in df.columns:
            if col.endswith(("_bias","_rmse")) or col in ("focal_mean_error","focal_sd_error"):
                if len(good) and col in good:r["mean_"+col]=float(pd.to_numeric(good[col]).mean())
        rows.append(r)
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--truth",required=True)
    ap.add_argument("--n",type=int,nargs="+",default=[500,1000,2500,5000])
    ap.add_argument("--reps",type=int,default=100)
    ap.add_argument("--seed",type=int,default=20261022)
    ap.add_argument("--starts",type=int,default=1)
    ap.add_argument("--out",default="mc_b12_7")
    a=ap.parse_args()
    if not 1<=a.reps<=10000:ap.error("Invalid repetitions.")
    truth=load_truth(a.truth);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    config={"n":a.n,"reps":a.reps,"seed":a.seed,"starts":a.starts,
            "conditions":["independent","testlet"],"truth":str(a.truth)}
    (out/"config.json").write_text(json.dumps(config,indent=2),encoding="utf8")
    dest=out/"replications.csv"
    old=pd.read_csv(dest) if dest.exists() else pd.DataFrame()
    seen=set(zip(old.N,old.rep,old.condition)) if not old.empty else set()
    for n in a.n:
        for cond in ("independent","testlet"):
            for rep in range(1,a.reps+1):
                if (n,rep,cond) in seen:continue
                row=run_one(n,rep,cond,truth,a.seed,a.starts)
                old=pd.concat([old,pd.DataFrame([row])],ignore_index=True)
                old.to_csv(dest,index=False)
                print(f"N={n} {cond} rep={rep}/{a.reps} status={row['status']} convergence={row['converged']}",flush=True)
    summarize(old).to_csv(out/"summary.csv",index=False)

if __name__=="__main__":main()
