"""Cold (fresh process incl. imports/quadrature) and warm runtime of closure-K sets, frozen vs optimised."""
import json, subprocess, sys, time
import numpy as np
CODE = r'''
import time, sys, importlib.util, tracemalloc
t0 = time.perf_counter()
import numpy as np
from validation.binary_certification.common import simulate
from core.irt_python_mirt_adapter import fit_python_mirt
mod = sys.argv[1]; r = int(sys.argv[2])
if mod == "frozen":
    spec = importlib.util.spec_from_file_location("F", "validation/p1_4pl_score_perf/frozen_k.py"); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
else:
    from core import irt_4pl_score_sets as M
x = simulate("4PL", 3000, 12, [20261126, 4, 100 + r, 0])
f = fit_python_mirt(x, model="4PL", estimation="MAP", n_quadpts=61, max_iter=500, tol=1e-4).fit
eta = f.coef()[["discrimination", "difficulty", "guessing", "upper"]].to_numpy(float).ravel()
t1 = time.perf_counter(); tracemalloc.start(); M.confidence_sets(eta, x); cold = time.perf_counter() - t1; peak = tracemalloc.get_traced_memory()[1] / 2**20; tracemalloc.stop()
t2 = time.perf_counter(); M.confidence_sets(eta, x); warm = time.perf_counter() - t2
print(cold, warm, peak, t1 - t0)
'''
res = {}
for mod in sys.argv[2].split(","):
    rows = [list(map(float, subprocess.run([sys.executable, "-c", CODE, mod, str(r)], capture_output=True, text=True, check=True).stdout.split())) for r in range(int(sys.argv[3]))]
    a = np.array(rows)
    res[mod] = dict(cold_median_s=float(np.median(a[:, 0])), cold_p95_s=float(np.percentile(a[:, 0], 95)), warm_median_s=float(np.median(a[:, 1])),
                    warm_p95_s=float(np.percentile(a[:, 1], 95)), peak_MiB=float(a[:, 2].max()), import_fit_median_s=float(np.median(a[:, 3])), reps=len(a))
json.dump(res, open(sys.argv[1], "w"), indent=1); print(json.dumps(res, indent=1))
