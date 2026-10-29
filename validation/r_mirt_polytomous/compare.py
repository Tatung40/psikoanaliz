"""Independent R mirt 1.48 comparison of unidimensional GRM, PCM, GPCM and NRM (Python mirt 1.2.0).

Validation only; no production code path. Each engine maximizes its own fine quadrature (Python 61-node
Gauss-Hermite, R 101 equally spaced points on [-6, 6]); comparisons are made on common, independent grids:
category probabilities on an 81-point theta grid and the marginal log-likelihood under a 121-node
Gauss-Hermite rule evaluated here for both parameter sets. The R -> Python mapping is first checked against
R's own probtrace at the fitted R parameters. Preregistered criteria: validation/mirt_stage1/criteria.json
(unchanged). R must actually run (Rscript); otherwise the report says NOT VERIFIED and the process fails.

    python validation/r_mirt_polytomous/compare.py --output validation-output/r-mirt-polytomous
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
MODELS = ("GRM", "PCM", "GPCM", "NRM")
THETA = np.linspace(-4, 4, 81)


def probabilities(model, params, theta):
    """(items, theta, categories) probabilities in Python mirt's physical parameterization."""
    theta = np.asarray(theta, float)
    if model == "GRM":
        a = np.asarray(params["discrimination"], float)[:, None, None]
        b = np.asarray(params["thresholds"], float)[:, None, :]
        star = expit(a * (theta[None, :, None] - b))
        ones = np.ones(star.shape[:2] + (1,))
        upper = np.concatenate([ones, star], axis=2)
        lower = np.concatenate([star, np.zeros_like(ones)], axis=2)
        return upper - lower
    if model in ("PCM", "GPCM"):
        a = np.asarray(params["discrimination"], float)[:, None, None]
        b = np.asarray(params["steps"], float)[:, None, :]
        steps = a * (theta[None, :, None] - b)
        z = np.concatenate([np.zeros(steps.shape[:2] + (1,)), np.cumsum(steps, axis=2)], axis=2)
    else:
        s = np.asarray(params["slopes"], float)[:, None, :]
        d = np.asarray(params["intercepts"], float)[:, None, :]
        z = s * theta[None, :, None] + d
    return np.exp(z - logsumexp(z, axis=2, keepdims=True))


def from_r(model, coef):
    """R mirt slope-intercept coefficients (one row per item) -> Python physical parameters."""
    a = np.asarray(coef["a1"], float)
    if model == "GRM":
        d = np.column_stack([coef[f"d{k}"] for k in range(1, 4)])
        return {"discrimination": a, "thresholds": -d / a[:, None]}
    if model in ("PCM", "GPCM"):
        ak = np.column_stack([coef[f"ak{k}"] for k in range(4)])
        d = np.column_stack([coef[f"d{k}"] for k in range(4)])
        if not np.allclose(ak, np.arange(4)) or not np.allclose(d[:, 0], 0):
            raise ValueError("R gpcm scoring/identification is not the default 0,1,2,3 with d0 = 0")
        return {"discrimination": a, "steps": -np.diff(d, axis=1) / a[:, None]}
    ak = np.column_stack([coef[f"ak{k}"] for k in range(4)])
    d = np.column_stack([coef[f"d{k}"] for k in range(4)])
    return {"slopes": a[:, None] * ak, "intercepts": d}


def common_loglik(model, params, x, nodes=121):
    z, w = hermegauss(nodes)
    p = np.clip(probabilities(model, params, z), 1e-300, 1)
    log_lik = np.zeros((x.shape[0], len(z)))
    for j in range(x.shape[1]):
        log_lik += np.log(p[j][:, x[:, j]]).T
    return float(logsumexp(log_lik + np.log(w / w.sum()), axis=1).sum())


def example(model):
    sys.path.insert(0, str(ROOT))
    from core import example_datasets as ex
    from core.irt_b14 import prepare_polytomous
    spec = next(d for d in ex.registry().values() if d.family == "irt" and d.model == model and d.level == 0)
    bundle = ex.materialize(spec.id)
    coded, _ = prepare_polytomous(bundle["data"][bundle["items"]], model,
                                  categories=["A", "B", "C", "D"] if model == "NRM" else None)
    return spec.id, coded.astype(int)


def python_fit(model, x):
    from core.irt_python_mirt_adapter import fit_python_mirt
    import mirt
    fit = fit_python_mirt(x, model=model, n_categories=4, n_quadpts=61, tol=1e-8, max_iter=5000).fit
    scores = mirt.fscores(fit, x, method="EAP", n_quadpts=61)
    params = {k: np.asarray(v, float) for k, v in fit.model.parameters.items()}
    return fit, params, np.asarray(scores.theta, float).ravel(), np.asarray(scores.standard_error, float).ravel()


def compare(model, out, criteria):
    case, data = example(model)
    folder = out / model
    folder.mkdir(parents=True, exist_ok=True)
    data.to_csv(folder / "data.csv", index=False)
    np.savetxt(folder / "theta.csv", THETA, delimiter=",", header="theta", comments="")
    run = subprocess.run(["Rscript", str(ROOT / "validation/r_mirt_polytomous/reference.R"), str(folder / "data.csv"),
                          model, str(folder)], capture_output=True, text=True)
    (folder / "stderr.log").write_text(run.stderr)
    if run.returncode:
        return {"model": model, "case": case, "status": "FAIL", "reason": "Rscript failed", "stderr": run.stderr[-2000:]}
    r = json.loads((folder / "r_result.json").read_text())
    x = data.to_numpy()
    fit, py, eap, psd = python_fit(model, x)
    rp = from_r(model, r["coef"])
    r_trace = np.asarray(r["probtrace"], float)            # items x theta x categories, from R itself
    mapping = float(np.abs(probabilities(model, rp, THETA) - r_trace).max())
    names = [k for k in py if not (model == "PCM" and k == "discrimination")]
    parameter = max(float(np.abs(py[k] - rp[k]).max()) for k in names)
    probability = float(np.abs(probabilities(model, py, THETA) - probabilities(model, rp, THETA)).max())
    loglik = abs(common_loglik(model, py, x) - common_loglik(model, rp, x))
    report = {"model": model, "case": case, "N": int(x.shape[0]), "items": int(x.shape[1]),
              "python_converged": bool(fit.converged), "R_converged": bool(r["converged"]),
              "python_loglik_native": float(fit.log_likelihood), "R_loglik_native": float(r["loglik"]),
              "python_parameters": int(fit.n_parameters), "R_parameters": int(r["nest"]),
              "mapping_probability": mapping, "calibration_parameters": parameter,
              "calibration_probability": probability, "common_logLik": loglik,
              "calibration_EAP": float(np.abs(eap - np.asarray(r["eap"], float)).max()),
              "calibration_posterior_SD": float(np.abs(psd - np.asarray(r["eap_se"], float)).max())}
    checks = {k: report[k] <= criteria[k] for k in ("mapping_probability", "calibration_parameters",
              "calibration_probability", "common_logLik", "calibration_EAP", "calibration_posterior_SD")}
    checks["converged"] = report["python_converged"] and report["R_converged"]
    checks["free_parameters"] = report["python_parameters"] == report["R_parameters"]
    report["checks"] = checks
    report["status"] = "NUMERICAL AGREEMENT / NOT CERTIFIED" if all(checks.values()) else "FAIL"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    args = parser.parse_args(argv)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if shutil.which("Rscript") is None:
        (out / "report.json").write_text(json.dumps({"status": "NOT VERIFIED", "reason": "Rscript unavailable"}))
        print("NOT VERIFIED: Rscript unavailable"); return 2
    criteria = json.loads((ROOT / "validation/mirt_stage1/criteria.json").read_text())
    reports = [compare(m, out, criteria) for m in args.models]
    (out / "report.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False))
    for r in reports:
        print("R_POLYTOMOUS=" + json.dumps(r, ensure_ascii=False))
    return 0 if all(r["status"].startswith("NUMERICAL AGREEMENT") for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
