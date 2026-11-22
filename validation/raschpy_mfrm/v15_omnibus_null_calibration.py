"""Independent parametric-null Monte Carlo calibration of RaschPy V15 omnibus LR.
300 calibration and 300 held-out null replications, disjoint deterministic seeds.
Not a conditional bootstrap for a particular observed dataset.
"""
import json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.stats import chi2, binomtest
from validation.raschpy_mfrm.simulation_v15 import one, ALPHA

def main():
    jobs=[("null",r) for r in range(3000,3600)]
    with Pool(2) as pool:
        rows=list(pool.imap_unordered(one,jobs))
    rows.sort(key=lambda x:x["rep"])
    result={"design":"independent parametric null Monte Carlo; 300 calibration / 300 held-out", "replications":len(rows)}
    for test in ("dif","drf"):
        values=np.array([r.get(test+"_om_lr",np.nan) for r in rows],dtype=float)
        dfs=np.array([r.get(test+"_om_df",np.nan) for r in rows],dtype=float)
        if not np.isfinite(values).all() or not np.isfinite(dfs).all() or len(set(dfs))!=1:
            result[test]={"status":"NOT CERTIFIED","error":"missing/nonfinite LR or inconsistent df"}
            continue
        calibration,heldout=values[:300],values[300:]
        cutoff=float(np.quantile(calibration,0.95,method="higher"))
        # Conservative rank p-value using independent calibration sample.
        p_emp=np.array([(1+np.sum(calibration>=v))/(len(calibration)+1) for v in heldout])
        p_chi=chi2.sf(heldout,dfs[0])
        n=len(heldout)
        emp_k=int(np.sum(p_emp<ALPHA))
        chi_k=int(np.sum(p_chi<ALPHA))
        emp_ci=binomtest(emp_k,n).proportion_ci(method="exact")
        chi_ci=binomtest(chi_k,n).proportion_ci(method="exact")
        # Strict rank-p boundary: p=(1+#calibration LR >= observed LR)/(m+1)<alpha.
        # Compute observed rejection directly; quantile cutoff is only descriptive.
        rank_p_min=1/(len(calibration)+1)
        result[test]={"df":float(dfs[0]),"calibration_n":300,"heldout_n":300,
            "chi_square_size":float(np.mean(p_chi<ALPHA)),
            "chi_square_rejections":chi_k,"chi_square_exact_95pct_ci":[chi_ci.low,chi_ci.high],
            "empirical_size":float(np.mean(p_emp<ALPHA)),
            "empirical_rejections":emp_k,"empirical_exact_95pct_ci":[emp_ci.low,emp_ci.high],
            "minimum_attainable_rank_p":rank_p_min,
            "empirical_95pct_cutoff":cutoff,
            "chi_square_95pct_cutoff":float(chi2.isf(ALPHA,dfs[0])),
            "heldout_lr":heldout.tolist(),
            "calibration_lr":calibration.tolist(),
            "status":"DIAGNOSTIC ONLY; not a replacement for preregistered V15 gates"}
    path=Path("validation/raschpy_mfrm/v15_omnibus_null_calibration.json")
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:{z:v for z,v in x.items() if z not in ("heldout_lr","calibration_lr")} if isinstance(x,dict) else x for k,x in result.items()},indent=2))

if __name__=="__main__": main()
