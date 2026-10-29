"""Pre-registered Monte Carlo: bifactor 2PL parameter recovery on the application path (core.irt_multidim.fit).

DESIGN and CRITERIA are fixed before any replication and written into the evidence; never adjusted to results.
Truth: 12 items, 3 specific groups of 4; general a = linspace(1.0, 1.8), specific a = (.6, .8, 1.0, 1.2) per
group, intercept d = linspace(-1, 1); theta ~ N(0, I4). Independent numpy generator. Fit: screen defaults for
4 dimensions (5 GH nodes per axis, max_iter 1000, tol 1e-7). Sign orientation is the application's own
(_canonical, positive reference loadings) and is compared with the positive truth directly.

    PYTHONPATH=. python validation/bifactor_recovery_study.py --output validation/evidence/p8_bifactor/recovery.json
"""
import argparse
import json
import platform
import time

import numpy as np

DESIGN = {"N": [500, 1000], "replications": 100, "items": 12, "groups": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
          "seed": 20261031, "general": "linspace(1.0, 1.8, 12)", "specific": "(.6, .8, 1.0, 1.2) per group",
          "intercept": "linspace(-1, 1, 12)", "fit": "core.irt_multidim.fit(..., 'Bifactor', groups, n_quadpts=5)"}
CRITERIA = {"convergence_rate_min": 0.95, "rmse_max": 0.30, "abs_mean_bias_max": 0.10,
            "rationale": "Declared before running; RMSE limit as for 3PL/4PL a/b usability (irt_3pl4pl_root_cause)."}


def truth():
    k = DESIGN["items"]
    return np.linspace(1, 1.8, k), np.tile([.6, .8, 1., 1.2], 3), np.linspace(-1, 1, k)


def run():
    import pandas as pd
    from scipy.special import expit
    from core.irt_multidim import fit
    aG, aS, d = truth(); g = np.asarray(DESIGN["groups"]); k = len(g)
    out = {}
    for index, n in enumerate(DESIGN["N"]):
        rng = np.random.default_rng([DESIGN["seed"], index])
        errors = {"general": [], "specific": [], "intercept": []}; converged = failed = 0
        for _ in range(DESIGN["replications"]):
            th = rng.standard_normal((n, 4))
            p = expit(aG * th[:, [0]] + aS * th[:, 1 + g] + d)
            x = pd.DataFrame((rng.random(p.shape) < p).astype(int), columns=[f"M{j}" for j in range(k)])
            try:
                f = fit(x, "Bifactor", g, n_quadpts=5)["fit"]
            except ValueError:
                failed += 1; continue
            converged += bool(f.converged)
            P = f.model.parameters
            for name, est, t in (("general", P["general_loadings"], aG), ("specific", P["specific_loadings"], aS),
                                 ("intercept", P["intercepts"], d)):
                errors[name].append(np.asarray(est, float) - t)
        res = {"replications": DESIGN["replications"], "converged": converged, "failed": failed}
        for name, e in errors.items():
            e = np.vstack(e)
            res[name] = {"mean_bias": float(e.mean()), "max_item_abs_bias": float(np.abs(e.mean(0)).max()),
                         "rmse": float(np.sqrt((e ** 2).mean())), "rmse_mc_se": float(np.sqrt((e ** 2).mean(1)).std(ddof=1) / np.sqrt(len(e)))}
        out[f"N{n}"] = res
    return out


def judge(results, criteria=CRITERIA):
    checks = {}
    for cell, r in results.items():
        checks[f"{cell}_convergence"] = r["converged"] / r["replications"] >= criteria["convergence_rate_min"]
        for name in ("general", "specific", "intercept"):
            checks[f"{cell}_{name}"] = r[name]["rmse"] <= criteria["rmse_max"] and abs(r[name]["mean_bias"]) <= criteria["abs_mean_bias_max"]
    return checks


def main(argv=None):
    import mirt
    parser = argparse.ArgumentParser(); parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    start = time.time(); results = run(); checks = judge(results)
    evidence = {"design": DESIGN, "criteria": CRITERIA, "results": results, "checks": checks,
                "status": "MET" if all(checks.values()) else "NOT MET", "wall_seconds": round(time.time() - start, 1),
                "versions": {"mirt": mirt.__version__, "numpy": np.__version__, "python": platform.python_version()}}
    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(evidence, indent=2, ensure_ascii=False))
    print(json.dumps({"status": evidence["status"], "checks": checks}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
