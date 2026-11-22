"""Independent parametric-null Monte Carlo calibration of RaschPy V15 omnibus LR.
60 calibration and 60 held-out null replications, disjoint deterministic seeds.
Not a conditional bootstrap for a particular observed dataset.
"""
import json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.stats import chi2
from validation.raschpy_mfrm.simulation_v15 import one, ALPHA

def main():
    jobs=[("null",r) for r in range(1000,1120)]
    with Pool(2) as pool:
        rows=list(pool.imap_unordered(one,jobs))
    rows.sort(key=lambda x:x["rep"])
    result={"design":"independent parametric null Monte Carlo; 60 calibration / 60 held-out", "replications":len(rows)}
    for test in ("dif","drf"):
        values=np.array([r.get(test+"_om_lr",np.nan) for r in rows],dtype=float)
        dfs=np.array([r.get(test+"_om_df",np.nan) for r in rows],dtype=float)
        if not np.isfinite(values).all() or not np.isfinite(dfs).all() or len(set(dfs))!=1:
            result[test]={"status":"NOT CERTIFIED","error":"missing/nonfinite LR or inconsistent df"}
            continue
        calibration,heldout=values[:60],values[60:]
        cutoff=float(np.quantile(calibration,0.95,method="higher"))
        # Conservative rank p-value using independent calibration sample.
        p_emp=np.array([(1+np.sum(calibration>=v))/(len(calibration)+1) for v in heldout])
        p_chi=chi2.sf(heldout,dfs[0])
        result[test]={"df":float(dfs[0]),"calibration_n":60,"heldout_n":60,
            "chi_square_size":float(np.mean(p_chi<ALPHA)),
            "empirical_size":float(np.mean(p_emp<ALPHA)),
            "empirical_95pct_cutoff":cutoff,
            "chi_square_95pct_cutoff":float(chi2.isf(ALPHA,dfs[0])),
            "heldout_lr":heldout.tolist(),
            "calibration_lr":calibration.tolist(),
            "status":"DIAGNOSTIC ONLY; not a replacement for preregistered V15 gates"}
    path=Path("validation/raschpy_mfrm/v15_omnibus_null_calibration.json")
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:{z:v for z,v in x.items() if z not in ("heldout_lr","calibration_lr")} if isinstance(x,dict) else x for k,x in result.items()},indent=2))

if __name__=="__main__": main()
