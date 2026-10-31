"""Pre-registered Monte Carlo: 1PL/2PL parameter recovery and Wald CI coverage on the application path.

Validates the item-parameter SE/CI shown on the Python mirt binary screens (core.irt_binary_inference),
not a separate estimator. Generator: independent numpy (not mirt.simdata). Fit: fit_python_mirt with the
screen defaults (21 GH nodes, tol 1e-4). CRITERIA and DESIGN are fixed before any replication is run and are
written into the evidence file; they are never adjusted to results.

    python validation/binary_se_coverage_study.py --output validation/evidence/p6_binary_se/coverage.json
"""
import argparse
import json
import platform
import time

import numpy as np

from validation.coverage_precision import coverage_precision

# CRITERIA below is the immutable historical preregistration. Its iid-interval
# MCSE rationale is not justified: items share respondents and calibration.
# New runs report replication-cluster MCSE without altering thresholds/results.

DESIGN = {"models": ["1PL", "2PL"], "replications": 300, "N": 1000, "items": 12, "seed": 20261029,
          "a": "linspace(0.8, 2.0, 12) for 2PL; 1 for 1PL", "b": "linspace(-1.5, 1.5, 12)",
          "theta": "N(0, 1)", "fit": "fit_python_mirt defaults (ML, 21 GH nodes, tol 1e-4, max_iter 500)",
          "ci": "95% Wald from core.irt_binary_inference.parameter_inference"}
CRITERIA = {"convergence_rate_min": 0.99,
            "abs_mean_bias_max": 0.05,
            "coverage_abs_error_max": 0.02,
            "rationale": ("Declared before running. Pooled over items per parameter type; with 300 x 12 intervals "
                          "the Monte Carlo SE of a 95% coverage is about 0.004, so 0.02 is about five MC SE. "
                          "Bias limit 0.05 logit is below one tenth of the smallest SE at N = 1000.")}


def simulate(rng, a, b, n):
    theta = rng.standard_normal(n)
    p = 1.0 / (1.0 + np.exp(-a[None, :] * (theta[:, None] - b[None, :])))
    return (rng.random(p.shape) < p).astype(int)


def run(design=DESIGN):
    from core.irt_python_mirt_adapter import fit_python_mirt
    from core.irt_binary_inference import parameter_inference
    k = design["items"]
    b_true = np.linspace(-1.5, 1.5, k)
    results = {}
    for model in design["models"]:
        a_true = np.ones(k) if model == "1PL" else np.linspace(0.8, 2.0, k)
        rng = np.random.default_rng([design["seed"], design["models"].index(model)])
        rows = {"a": [], "b": []}
        converged = withheld = constant_item_skipped = 0
        for _ in range(design["replications"]):
            x = simulate(rng, a_true, b_true, design["N"])
            if (x.mean(0) == 0).any() or (x.mean(0) == 1).any():
                constant_item_skipped += 1
                continue
            fit = fit_python_mirt(x, model=model).fit
            converged += bool(fit.converged)
            table, reason = parameter_inference(fit, model, [f"M{j}" for j in range(k)])
            if table is None:
                withheld += 1
                continue
            for name, truth in (("a", a_true), ("b", b_true)):
                part = table[table.Parametre == name]
                if part.empty:
                    continue
                est = part["Kestirim"].to_numpy(); low = part["GA %95 alt"].to_numpy(); high = part["GA %95 üst"].to_numpy()
                rows[name].append(np.column_stack([est - truth, (low <= truth) & (truth <= high)]))
        summary = {"replications": design["replications"], "converged": converged, "se_withheld": withheld,
                   "constant_item_skipped": constant_item_skipped}
        for name, parts in rows.items():
            if not parts:
                continue
            arr = np.vstack(parts); n = len(arr)
            coverage = float(arr[:, 1].mean())
            summary[name] = {"intervals": n, "mean_bias": float(arr[:, 0].mean()),
                             "max_item_abs_bias": float(np.abs(np.mean(np.stack(parts)[:, :, 0], axis=0)).max()),
                             "rmse": float(np.sqrt((arr[:, 0] ** 2).mean())), "coverage": coverage,
                             **coverage_precision(parts, design["replications"])}
        results[model] = summary
    return results


def judge(results, criteria=CRITERIA):
    checks = {}
    for model, s in results.items():
        checks[f"{model}_convergence"] = s["converged"] / s["replications"] >= criteria["convergence_rate_min"]
        for name in ("a", "b"):
            if name in s:
                checks[f"{model}_{name}_bias"] = abs(s[name]["mean_bias"]) <= criteria["abs_mean_bias_max"]
                checks[f"{model}_{name}_coverage"] = abs(s[name]["coverage"] - .95) <= criteria["coverage_abs_error_max"]
    return checks


def main(argv=None):
    import mirt
    parser = argparse.ArgumentParser(); parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    start = time.time()
    results = run()
    checks = judge(results)
    evidence = {"design": DESIGN, "criteria": CRITERIA, "results": results, "checks": checks,
                "status": "MET (pre-registered numerical criteria; not a general certification)" if all(checks.values())
                else "NOT MET", "wall_seconds": round(time.time() - start, 1),
                "versions": {"mirt": mirt.__version__, "numpy": np.__version__, "python": platform.python_version()}}
    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(evidence, indent=2, ensure_ascii=False))
    print(json.dumps({"status": evidence["status"], "checks": checks}, ensure_ascii=False))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
