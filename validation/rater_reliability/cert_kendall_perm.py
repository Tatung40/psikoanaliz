"""R-A2 fresh validation for Kendall W permutation p."""
import json, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, "validation/rater_reliability")
from cert_run import C

def run(a):
    seed, ties = a; rng = np.random.default_rng(seed)
    return sum(C("kendall_w", (rng.integers(1, 5, (10, 5)) if ties else rng.normal(size=(10, 5))))["p"] < .05 for _ in range(250))

if __name__ == "__main__":
    ss = np.random.default_rng(52).integers(1e9, size=16)
    with ProcessPoolExecutor(16) as ex: r = list(ex.map(run, [(int(s), i >= 8) for i, s in enumerate(ss)]))
    t1, t2 = sum(r[:8]) / 2000, sum(r[8:]) / 2000
    x = np.random.default_rng(7).integers(1, 5, (12, 4)); a, b = C("kendall_w", x), C("kendall_w", x)
    from scipy import stats
    out = dict(type1=t1, type1_ties=t2, deterministic=a == b, p=a["p"], p_chi2=a["p_chi2"],
               passed=bool(.036 <= t1 <= .064 and t2 <= .064 and a == b and 0 < a["p"] <= 1))
    json.dump(out, open(sys.argv[1], "w"), indent=1, default=float); print(out)
