"""G2/G3 new cells of PROTOCOL.json: recovery and 95% Wald coverage on the application path.

    PYTHONPATH=. python validation/binary_certification/coverage.py --workers 2 --output validation/evidence/binary_certification/coverage.json
"""
import argparse
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROTOCOL = json.loads((Path(__file__).with_name("PROTOCOL.json")).read_text())
G3 = PROTOCOL["gates"]["G3_inference"]
REPS = {1000: G3["replications"]["N1000"], 3000: G3["replications"]["N3000"], 10000: G3["replications"]["N10000"]}
JOBS = [(label, n, k) for label, cells in G3["new_cells"].items() for n, k in cells]


def _cell(job):
    from core.irt_binary_inference import parameter_inference
    from core.irt_python_mirt_adapter import fit_python_mirt
    from validation.binary_certification.common import simulate, truth
    from validation.coverage_precision import coverage_precision
    label, n, k = job
    model, variant = label.split()[0], (label.split()[1] if " " in label else None)
    estimation = "MAP" if variant == "MAP" else "ML"
    variant = None if variant == "MAP" else variant
    p = truth(model, k, variant); names = [f"M{j}" for j in range(k)]
    reps = REPS[n]; index = JOBS.index(job)
    parts = {q: [] for q in ("a", "b", "c", "d")}; converged = refused = 0
    for r in range(reps):
        x = simulate(model, n, k, [G3["seed"], index, r], variant)
        fit = fit_python_mirt(x, model=model, estimation=estimation, common_discrimination=variant == "common_a").fit
        converged += bool(fit.converged)
        table, _ = parameter_inference(fit, model, names)
        if table is None:
            refused += 1
            continue
        for q in parts:
            part = table[table.Parametre == q]
            if part.empty or (q == "a" and variant == "fixed_a"):
                continue
            j = part.Madde.str[1:].astype(int).to_numpy(); t = np.asarray(p[q])[j]
            est = part.Kestirim.to_numpy(); low = part["GA %95 alt"].to_numpy(); high = part["GA %95 üst"].to_numpy()
            if q == "a" and variant == "common_a":      # one shared parameter: count it once per dataset
                est, low, high, t = est[:1], low[:1], high[:1], t[:1]
            parts[q].append(np.column_stack([est - t, np.isnan(low), (low <= t) & (t <= high)]))
    out = {"cell": label, "N": n, "k": k, "estimation": estimation, "replications": reps,
           "converged": converged, "se_refused": refused}
    for q, rows in parts.items():
        if rows:
            arr = np.vstack(rows); shown = arr[arr[:, 1] == 0]
            out[q] = {"intervals": int(len(arr)), "withheld_rate": float(arr[:, 1].mean()),
                      "mean_bias": float(arr[:, 0].mean()), "rmse": float(np.sqrt((arr[:, 0] ** 2).mean())),
                      "coverage": float(shown[:, 2].mean()) if len(shown) else float("nan"),
                      **coverage_precision(rows, reps, eligible_column=1)}
    return out


def judge(r):
    c3, c2 = G3["criteria"], PROTOCOL["gates"]["G2_recovery"]["new_cells_criteria"]
    checks = {"convergence": r["converged"] / r["replications"] >= c3["convergence_rate_min"],
              "se_refused": r["se_refused"] <= c3["se_refused"]}
    for q in ("a", "b", "c", "d"):
        if q in r:
            s = r[q]
            checks[f"{q}_coverage"] = abs(s["coverage"] - .95) <= c3["coverage_abs_error_max"]
            checks[f"{q}_withheld"] = s["withheld_rate"] <= c3["withheld_rate_max"]
            if q in "ab":
                checks[f"{q}_bias"] = abs(s["mean_bias"]) <= c2["abs_pooled_bias_a_b_max"]
                checks[f"{q}_rmse"] = s["rmse"] <= c2["rmse_a_b_max"]
            else:
                checks[f"{q}_rmse"] = s["rmse"] <= c2["rmse_c_d_max"]
    return checks


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=2); ap.add_argument("--output", required=True)
    a = ap.parse_args(argv); start = time.time()
    order = sorted(JOBS, key=lambda j: -j[1] * j[2])              # longest first
    with ProcessPoolExecutor(a.workers) as pool:
        results = dict(zip(order, pool.map(_cell, order)))
    results = [results[j] for j in JOBS]
    for r in results:
        r["checks"] = judge(r); r["pass"] = all(r["checks"].values())
    import mirt
    evidence = {"protocol": "validation/binary_certification/PROTOCOL.json (G2/G3 new cells)", "results": results,
                "all_pass": all(r["pass"] for r in results), "wall_seconds": round(time.time() - start, 1),
                "versions": {"mirt": mirt.__version__, "numpy": np.__version__, "python": platform.python_version()}}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(evidence, indent=1, allow_nan=True) + "\n")
    print(json.dumps({r["cell"] + f" N{r['N']} k{r['k']}": r["pass"] for r in results}, indent=1))


if __name__ == "__main__":
    main()
