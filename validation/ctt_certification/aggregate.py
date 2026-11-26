"""CTT-C1 gate evaluation (PROTOCOL.json). Refusals counted; nothing dropped silently."""
import glob, json, sys
import numpy as np
sys.path.insert(0, "validation/ctt_certification")
from run import cells, estimands

rows = [json.loads(l) for f in glob.glob(f"{sys.argv[1]}/*.jsonl") for l in open(f)]
C = cells(); rep = {"cells": [], "methods": {}}
for ci, c in enumerate(C):
    R = [r for r in rows if r["cell"] == ci]; _, est, _ = estimands(c); tol = .01 if c["N"] == 1000 else .02
    methods = ["alpha", "feldt_brennan", "feldt_gilmer"] + (["kr20"] if c["kind"] == "binary" else ["omega_std", "omega_raw"])
    cell = dict(cell=ci, **c, R=len(R), errors=sum("error" in r for r in R), results={})
    for m in methods:
        v = np.array([r.get(m) if r.get(m) is not None else np.nan for r in R], float); ok = np.isfinite(v); n = int(ok.sum())
        refusal = 1 - n / max(len(R), 1); bias = float(v[ok].mean() - est[m]) if n else np.nan; sd = float(v[ok].std(ddof=1)) if n > 1 else np.nan
        mcse = sd / np.sqrt(n) if n else np.nan; rmse = float(np.sqrt(np.mean((v[ok] - est[m]) ** 2))) if n else np.nan
        rmax = .05 if m.startswith("omega") else .01
        gates = {"bias_bound": bool(abs(bias) + 2.576 * mcse <= tol), "mcse<=.002": bool(mcse <= .002), "refusal": bool(refusal <= rmax), "R=1000": len(R) == 1000}
        cell["results"][m] = dict(estimand=est[m], mean=float(v[ok].mean()) if n else None, bias=bias, rmse=rmse, mcse=mcse, refusal=refusal, gates=gates, passed=all(gates.values()))
        rep["methods"].setdefault(m, []).append(cell["results"][m]["passed"])
    if "true_reliability" in est: cell["true_reliability"] = est["true_reliability"]
    rep["cells"].append(cell)
rep["status"] = {m: ("CERTIFIED" if all(p) else "PARTIAL" if any(p) else "NOT CERTIFIED") for m, p in rep["methods"].items()}
json.dump(rep, open(sys.argv[2], "w"), indent=1); print("CTT_C1 " + json.dumps(rep["status"])); print("CTT_C1_FULL " + json.dumps(rep))
