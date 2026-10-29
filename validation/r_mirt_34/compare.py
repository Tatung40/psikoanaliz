"""Independent R mirt 1.48 comparison of ML 3PL and 4PL, including observed-information SEs (Python mirt 1.2.0).

Validation only. Design (fixed before any R run):
  data      N = 5000, 15 items, independent numpy generator, truth of validation/irt_3pl4pl_root_cause.truth.
  fixture   SEEDS: for each model the first seed in 1..50 whose Python ML fit (61 GH, tol 1e-8) converges with no
            parameter on a library box bound (select_seeds()). Python holds bound parameters fixed while R's
            bounds are open intervals, so only interior solutions are the same constrained problem. The rule
            selects on Python's boundary status only, never on agreement with R.
  Python    fit_python_mirt ML, 61 GH nodes, tol 1e-8, max_iter 5000; SE = library Oakes observed information
            through core.irt_binary_inference.parameter_inference (the screen path).
  R         reference.R: mirt 3PL/4PL, ML, 101 points on [-6, 6], TOL 1e-8, SQUAREM, up to 100000 cycles,
            SE.type = "Oakes"; physical a, b, c, d and delta-method SEs from coef(IRTpars = TRUE, printSE = TRUE).
  criteria  validation/mirt_stage1/criteria.json (unchanged) for parameters, IRF, common logLik; IIF/TIF on the
            same 81-point grid with the calibration_probability tolerance applied relative to max TIF; SE: max
            relative difference <= .02 (declared now: Oakes by finite differences in both engines on different
            quadratures). R non-convergence is reported as NOT VERIFIED, never as a Python failure.

    python validation/r_mirt_34/compare.py --output validation-output/r-mirt-34
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import expit, logsumexp

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("3PL", "4PL")
N, K = 5000, 15
SEEDS = {}          # filled by select_seeds() before the first R run; see README
THETA = np.linspace(-4, 4, 81)
SE_RELATIVE_MAX = 0.02


def simulate(model, seed):
    sys.path.insert(0, str(ROOT))
    from validation.irt_3pl4pl_root_cause import truth
    p = truth(model, K)
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal(N)
    prob = p["c"] + (p["d"] - p["c"]) * expit(p["a"] * (theta[:, None] - p["b"]))
    return (rng.random(prob.shape) < prob).astype(int)


def python_fit(model, x):
    from core.irt_binary_inference import parameter_inference
    from core.irt_python_mirt_adapter import fit_python_mirt
    fit = fit_python_mirt(x, model=model, n_quadpts=61, tol=1e-8, max_iter=5000).fit
    table, reason = parameter_inference(fit, model, [f"V{j + 1}" for j in range(x.shape[1])])
    return fit, table, reason


def select_seeds():
    seeds = {}
    for model in MODELS:
        for seed in range(1, 51):
            fit, table, _ = python_fit(model, simulate(model, seed))
            if fit.converged and table is not None and (table.Not == "").all():
                seeds[model] = seed
                break
    return seeds


def params(table, model):
    out = {q: table[table.Parametre == q].Kestirim.to_numpy() for q in "abc"}
    out["d"] = table[table.Parametre == "d"].Kestirim.to_numpy() if model == "4PL" else np.ones(len(out["a"]))
    return out


def irf(p, theta):
    return p["c"] + (p["d"] - p["c"]) * expit(p["a"] * (theta[:, None] - p["b"]))


def information(p, theta):
    s = expit(p["a"] * (theta[:, None] - p["b"])); prob = irf(p, theta)
    slope = p["a"] * (p["d"] - p["c"]) * s * (1 - s)
    return slope ** 2 / (prob * (1 - prob))


def common_loglik(p, x, nodes=121):
    z, w = hermegauss(nodes)
    prob = irf(p, z)
    ll = x @ np.log(prob).T + (1 - x) @ np.log1p(-prob).T
    return float(logsumexp(ll + np.log(w / w.sum()), axis=1).sum())


def compare(model, out, criteria):
    seed = SEEDS.get(model)
    if seed is None:
        return {"model": model, "status": "NOT APPLICABLE", "reason": "no interior Python ML solution in seeds 1..50"}
    x = simulate(model, seed)
    folder = out / model; folder.mkdir(parents=True, exist_ok=True)
    header = ",".join(f"V{j + 1}" for j in range(K))
    np.savetxt(folder / "data.csv", x, fmt="%d", delimiter=",", header=header, comments="")
    run = subprocess.run(["Rscript", str(ROOT / "validation/r_mirt_34/reference.R"), str(folder / "data.csv"), model,
                          str(folder)], capture_output=True, text=True)
    (folder / "stderr.log").write_text(run.stderr)
    if run.returncode:
        return {"model": model, "seed": seed, "status": "FAIL", "reason": "Rscript failed", "stderr": run.stderr[-2000:]}
    r = json.loads((folder / "r_result.json").read_text())
    fit, table, reason = python_fit(model, x)
    if table is None or not (table.Not == "").all():
        return {"model": model, "seed": seed, "status": "FAIL", "reason": f"Python fixture no longer interior: {reason}"}
    py = params(table, model)
    rp = {q: np.asarray(r["par"][q], float) for q in "abcd"}
    r_se = {q: np.asarray(r["se"][q], float) for q in "abcd"}
    names = "abcd" if model == "4PL" else "abc"
    py_se = {q: table[table.Parametre == q].SH.to_numpy() for q in names}
    tif_py, tif_r = information(py, THETA).sum(1), information(rp, THETA).sum(1)
    report = {"model": model, "seed": seed, "N": N, "items": K,
              "python_converged": bool(fit.converged), "R_converged": bool(r["converged"]),
              "R_iterations": int(r["iterations"]),
              "python_loglik_native": float(fit.log_likelihood), "R_loglik_native": float(r["loglik"]),
              "calibration_parameters": max(float(np.abs(py[q] - rp[q]).max()) for q in names),
              "calibration_probability": float(np.abs(irf(py, THETA) - irf(rp, THETA)).max()),
              "common_logLik": abs(common_loglik(py, x) - common_loglik(rp, x)),
              "information_relative": float(np.abs(information(py, THETA) - information(rp, THETA)).max() / tif_r.max()),
              "tif_relative": float(np.abs(tif_py - tif_r).max() / tif_r.max()),
              "se_relative": max(float(np.abs(py_se[q] / r_se[q] - 1).max()) for q in names)}
    checks = {k: report[k] <= criteria[k] for k in ("calibration_parameters", "calibration_probability", "common_logLik")}
    checks["information"] = report["information_relative"] <= criteria["calibration_probability"]
    checks["tif"] = report["tif_relative"] <= criteria["calibration_probability"]
    checks["se"] = report["se_relative"] <= SE_RELATIVE_MAX
    report["checks"] = checks
    if not report["R_converged"]:
        report["status"] = "NOT VERIFIED (R did not converge)"
    else:
        report["status"] = "NUMERICAL AGREEMENT / NOT CERTIFIED" if all(checks.values()) and report["python_converged"] else "FAIL"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--select-seeds", action="store_true")
    args = parser.parse_args(argv)
    if args.select_seeds:
        print(json.dumps(select_seeds())); return 0
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if shutil.which("Rscript") is None:
        (out / "report.json").write_text(json.dumps({"status": "NOT VERIFIED", "reason": "Rscript unavailable"}))
        print("NOT VERIFIED: Rscript unavailable"); return 2
    criteria = json.loads((ROOT / "validation/mirt_stage1/criteria.json").read_text())
    reports = [compare(m, out, criteria) for m in MODELS]
    (out / "report.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False))
    for r in reports:
        print("R_3PL4PL=" + json.dumps(r, ensure_ascii=False))
    return 0 if all(r["status"].startswith(("NUMERICAL AGREEMENT", "NOT VERIFIED", "NOT APPLICABLE")) for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
