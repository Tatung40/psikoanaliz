"""Prespecified tolerances: alpha 1e-6 (production rounds to 6 dp), Guttman 1e-10 (closed form, unrounded),
omega 1e-4 (production rounds to 4 dp; two ML optimisers). Numerical agreement only."""
import json, sys
d = sys.argv[1]; P = json.load(open(f"{d}/python.json")); R = json.load(open(f"{d}/r.json")); rows = []; ok = True
for nm in P:
    p, r = P[nm], R[nm]
    checks = [("alpha", p["alpha"], r["alpha"], 1e-6), ("omega_std_ml", p["omega_std_ml"], r["omega_std_ml"], 1e-4),
              ("lambda1(cor)", p["cor"]["lambda1"], r["lambda1"], 1e-10), ("lambda2(cor)", p["cor"]["lambda2"], r["lambda2"], 1e-10),
              ("lambda3(cor)", p["cor"]["lambda3"], r["lambda3"], 1e-10), ("lambda6(cor)", p["cor"]["lambda6"], r["lambda6"], 1e-10)]
    for n, a, b, tol in checks:
        e = None if a is None or b is None else abs(a - b); good = e is not None and e <= tol; ok &= good
        rows.append(dict(fixture=nm, statistic=n, python=a, r=b, abs_error=e, tolerance=tol, agree=good))
    rows.append(dict(fixture=nm, statistic="lavaan_converged/heywood", python=None, r=[r["lavaan_converged"], r["heywood"]], agree=True))
rep = {"agree_all": ok, "rows": rows, "psych": R["poly"]["psych"], "lavaan": R["poly"]["lavaan"],
       "status": "NUMERICAL AGREEMENT CHECK — not scientific certification"}
json.dump(rep, open(f"{d}/COMPARISON.json", "w"), indent=1); print("CTT_REF " + json.dumps(rep)); sys.exit(0)
