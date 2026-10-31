"""Pre-registered Monte Carlo: 3PL/4PL Wald CI coverage of the SE shown on the Python mirt binary screens.

Application path only: fit_python_mirt (screen defaults) -> core.irt_binary_inference.parameter_inference.
Parameters on a library box bound receive no SE; they are counted as withheld, not as covered.
Truth: validation/irt_3pl4pl_root_cause.py (a .8-1.6, b -1.25..1.25, c .10-.18, d .90-.97); generator
independent numpy. DESIGN and CRITERIA are fixed before any replication and written into the evidence.

Decision rule (fixed now): a (model, estimation, parameter) gets a Wald CI on screen only if, in every cell
of DESIGN for that (model, estimation), convergence >= .99, withheld (on-bound) rate <= .05 and
|coverage - .95| <= .02. Otherwise the screen keeps the SE but shows NA for the CI with the reason.

    PYTHONPATH=. python validation/binary34_se_coverage_study.py --output validation/evidence/p7_binary34_se/coverage.json
"""
import argparse
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from validation.coverage_precision import coverage_precision

# Preserve historical DESIGN/CRITERIA and raw evidence; their iid-interval MCSE
# rationale is unverified. New runs use independent datasets as clusters.

CELLS = [("3PL", "MAP", 3000, 30), ("4PL", "MAP", 3000, 30), ("3PL", "ML", 3000, 30), ("4PL", "ML", 3000, 30),
         ("3PL", "ML", 10000, 30)]
DESIGN = {"cells": CELLS, "replications": {"N3000": 200, "N10000": 100}, "seed": 20261030,
          "truth": "validation/irt_3pl4pl_root_cause.truth", "theta": "N(0, 1)",
          "fit": "fit_python_mirt screen defaults (EM, 21 GH nodes, tol 1e-4, max_iter 500); MAP = library default priors",
          "ci": "95% Wald from core.irt_binary_inference.parameter_inference (no evidence gating)"}
CRITERIA = {"convergence_rate_min": 0.99, "withheld_rate_max": 0.05, "coverage_abs_error_max": 0.02,
            "rationale": ("Declared before running; same coverage tolerance as p6_binary_se. With >= 100 x 30 "
                          "intervals the MC SE of a 95% coverage is <= .004. Withheld intervals are excluded from "
                          "coverage, so a high withheld rate would make coverage a selected subset.")}


def _cell(args):
    from core.irt_binary_inference import parameter_inference
    from core.irt_python_mirt_adapter import fit_python_mirt
    from validation.irt_3pl4pl_root_cause import truth
    from scipy.special import expit
    model, estimation, n, k, reps, index = args
    p = truth(model, k)
    rng = np.random.default_rng([DESIGN["seed"], index])
    names = [f"M{j}" for j in range(k)]
    parts = {q: [] for q in ("a", "b", "c", "d")}
    converged = refused = 0
    for _ in range(reps):
        theta = rng.standard_normal(n)
        prob = p["c"] + (p["d"] - p["c"]) * expit(p["a"] * (theta[:, None] - p["b"]))
        x = (rng.random(prob.shape) < prob).astype(int)
        fit = fit_python_mirt(x, model=model, estimation=estimation).fit
        converged += bool(fit.converged)
        table, _ = parameter_inference(fit, model, names)
        if table is None:
            refused += 1
            continue
        for q in parts:
            part = table[table.Parametre == q]
            if part.empty:
                continue
            j = part.Madde.str[1:].astype(int).to_numpy(); t = p[q][j]
            est = part.Kestirim.to_numpy(); low = part["GA %95 alt"].to_numpy(); high = part["GA %95 üst"].to_numpy()
            parts[q].append(np.column_stack([est - t, np.isnan(low), (low <= t) & (t <= high)]))
    out = {"model": model, "estimation": estimation, "N": n, "items": k, "replications": reps,
           "converged": converged, "se_refused": refused}
    for q, rows in parts.items():
        if not rows:
            continue
        arr = np.vstack(rows); shown = arr[arr[:, 1] == 0]
        cov = float(shown[:, 2].mean()) if len(shown) else float("nan")
        out[q] = {"intervals": int(len(arr)), "withheld_rate": float(arr[:, 1].mean()),
                  "mean_bias": float(arr[:, 0].mean()), "rmse": float(np.sqrt((arr[:, 0] ** 2).mean())),
                  "coverage": cov,
                  **coverage_precision(rows, reps, eligible_column=1)}
    return out


def run(workers=5):
    jobs = [(m, e, n, k, DESIGN["replications"][f"N{n}"], i) for i, (m, e, n, k) in enumerate(CELLS)]
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(_cell, jobs))


def judge(results, criteria=CRITERIA):
    checks, ci = {}, {}
    for r in results:
        tag = f'{r["model"]}_{r["estimation"]}_N{r["N"]}'
        conv = r["converged"] / r["replications"] >= criteria["convergence_rate_min"] and r["se_refused"] == 0
        checks[f"{tag}_convergence"] = conv
        for q in "abcd":
            if q in r:
                ok = (conv and r[q]["withheld_rate"] <= criteria["withheld_rate_max"]
                      and abs(r[q]["coverage"] - .95) <= criteria["coverage_abs_error_max"])
                checks[f"{tag}_{q}"] = ok
                key = f'{r["model"]}_{r["estimation"]}_{q}'
                ci[key] = ci.get(key, True) and ok
    return checks, ci


def main(argv=None):
    import mirt
    parser = argparse.ArgumentParser(); parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    start = time.time()
    results = run()
    checks, ci = judge(results)
    evidence = {"design": DESIGN, "criteria": CRITERIA, "results": results, "checks": checks,
                "wald_ci_supported": ci, "wall_seconds": round(time.time() - start, 1),
                "versions": {"mirt": mirt.__version__, "numpy": np.__version__, "python": platform.python_version()}}
    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(evidence, indent=2, ensure_ascii=False))
    print(json.dumps(ci))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
