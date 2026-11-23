"""Sensitivity-only null DGP check: two sample sizes, distinct seeds.
No certification; preregistered V15 thresholds remain unchanged.
"""
import json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.stats import chi2, binomtest
import validation.raschpy_mfrm.simulation_v15 as sim

def worker_init(n):
    sim.N=n

def scenario(n, start, reps=40):
    jobs=[("null",r) for r in range(start,start+reps)]
    with Pool(2,initializer=worker_init,initargs=(n,)) as pool:
        rows=list(pool.imap_unordered(sim.one,jobs))
    result={"N":n,"replications":len(rows),"seed_reps":[start,start+reps-1]}
    for test in ("dif","drf"):
        lr=np.array([r.get(test+"_om_lr",np.nan) for r in rows],float)
        df=np.array([r.get(test+"_om_df",np.nan) for r in rows],float)
        if len(lr)!=reps or not np.isfinite(lr).all() or not np.isfinite(df).all() or len(set(df))!=1:
            result[test]={"status":"INVALID","errors":[r.get(test+"_error") for r in rows if r.get(test+"_error")]}
            continue
        k=int(np.sum(chi2.sf(lr,df[0])<.05))
        ci=binomtest(k,reps).proportion_ci(method="exact")
        result[test]={"chi2_rejections":k,"size":k/reps,"exact_95pct_ci":[ci.low,ci.high],
                      "lr_q95":float(np.quantile(lr,.95)),"df":float(df[0]),
                      "status":"SENSITIVITY ONLY"}
    return result

def main():
    result={"design":"two independent null DGP sample-size sensitivities; no recalibration or certification",
            "scenarios":[scenario(250,4000),scenario(750,5000)]}
    path=Path("validation/raschpy_mfrm/v15_omnibus_sensitivity.json")
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
