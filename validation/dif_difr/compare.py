"""Prespecified tolerances (PROTOCOL.json); per fixture and statistic. Numerical agreement only."""
import json, sys
import numpy as np
P = json.load(open(sys.argv[1])); R = json.load(open(sys.argv[2])); T = json.load(open("validation/dif_difr/PROTOCOL.json"))["tolerances_v2"]
rows = []
for f, p in P.items():
    for k, tol in T.items():
        a, b = np.asarray(p[k], float), np.asarray(R[f][k], float); e = float(np.nanmax(np.abs(a - b))) if a.shape == b.shape else float("inf")
        rows.append(dict(fixture=f, statistic=k, max_abs_error=e, tolerance=tol, agree=bool(e <= tol and np.array_equal(np.isnan(a), np.isnan(b)))))
rep = {"versions": R["versions"], "rows": rows, "agree_all": all(r["agree"] for r in rows), "status": "NUMERICAL AGREEMENT ONLY"}
print("DIFR " + json.dumps(rep))
