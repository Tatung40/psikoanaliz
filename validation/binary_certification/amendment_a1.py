"""Amendment A1 of PROTOCOL.json: A1a (1PL common_a precision) and A1b (MAP sandwich SE), registered before running.

    PYTHONPATH=. python validation/binary_certification/amendment_a1.py --workers 2 --output validation/evidence/binary_certification/amendment_a1.json
"""
import argparse
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from validation.binary_certification.coverage import judge

A1 = json.loads((Path(__file__).with_name("PROTOCOL.json")).read_text())["amendment_A1"]
JOBS = ([("1PL common_a", n, k, 1000, 20261125, "posterior") for n in (1000, 10000) for k in (12, 30)]
        + [(f"{m} MAP", n, k, 300 if n == 3000 else 150, 20261126, "sandwich")
           for m in ("3PL", "4PL") for n in (3000, 10000) for k in (12, 30)])


def _cell(job):
    from core.irt_binary_inference import parameter_inference
    from core.irt_python_mirt_adapter import fit_python_mirt
    from validation.binary_certification.common import simulate, truth
    from validation.coverage_precision import coverage_precision
    label, n, k, reps, seed, se_method = job
    model, variant = label.split()
    estimation = "MAP" if variant == "MAP" else "ML"; variant = None if variant == "MAP" else variant
    p = truth(model, k, variant); names = [f"M{j}" for j in range(k)]; index = JOBS.index(job)
    parts = {q: [] for q in ("a", "b", "c", "d")}; converged = refused = 0
    for r in range(reps):
        x = simulate(model, n, k, [seed, index, r], variant)
        fit = fit_python_mirt(x, model=model, estimation=estimation, common_discrimination=variant == "common_a").fit
        converged += bool(fit.converged)
        table, _ = parameter_inference(fit, model, names, se_method=se_method)
        if table is None:
            refused += 1; continue
        for q in parts:
            part = table[table.Parametre == q]
            if part.empty:
                continue
            j = part.Madde.str[1:].astype(int).to_numpy(); t = np.asarray(p[q])[j]
            est = part.Kestirim.to_numpy(); low = part["GA %95 alt"].to_numpy(); high = part["GA %95 üst"].to_numpy()
            if q == "a" and variant == "common_a":
                est, low, high, t = est[:1], low[:1], high[:1], t[:1]
            parts[q].append(np.column_stack([est - t, np.isnan(low), (low <= t) & (t <= high)]))
    out = {"cell": label, "N": n, "k": k, "estimation": estimation, "se_method": se_method, "replications": reps,
           "converged": converged, "se_refused": refused}
    for q, rows in parts.items():
        if rows:
            arr = np.vstack(rows); shown = arr[arr[:, 1] == 0]
            out[q] = {"intervals": int(len(arr)), "withheld_rate": float(arr[:, 1].mean()),
                      "mean_bias": float(arr[:, 0].mean()), "rmse": float(np.sqrt((arr[:, 0] ** 2).mean())),
                      "coverage": float(shown[:, 2].mean()) if len(shown) else float("nan"),
                      **coverage_precision(rows, reps, eligible_column=1)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=2); ap.add_argument("--output", required=True)
    a = ap.parse_args(argv); start = time.time()
    order = sorted(JOBS, key=lambda j: -j[1] * j[2] * j[3])
    with ProcessPoolExecutor(a.workers) as pool:
        done = dict(zip(order, pool.map(_cell, order)))
    results = [done[j] for j in JOBS]
    for r in results:
        r["checks"] = judge(r)
        if r["se_method"] == "sandwich":
            r["checks"]["se_refused_rate"] = r["se_refused"] / r["replications"] <= 0.01
            r["checks"]["se_refused"] = True       # A1b replaces the zero-refusal rule by the registered <= .01 rate
        r["pass"] = all(r["checks"].values())
    import mirt
    ev = {"protocol": "validation/binary_certification/PROTOCOL.json amendment_A1", "results": results,
          "A1a_pass": all(r["pass"] for r in results if r["cell"] == "1PL common_a"),
          "A1b_pass": {m: all(r["pass"] for r in results if r["cell"] == f"{m} MAP") for m in ("3PL", "4PL")},
          "wall_seconds": round(time.time() - start, 1),
          "versions": {"mirt": mirt.__version__, "numpy": np.__version__, "python": platform.python_version()}}
    Path(a.output).write_text(json.dumps(ev, indent=1, allow_nan=True) + "\n")
    print(json.dumps({f"{r['cell']} N{r['N']} k{r['k']}": r["pass"] for r in results}, indent=1))


if __name__ == "__main__":
    main()
