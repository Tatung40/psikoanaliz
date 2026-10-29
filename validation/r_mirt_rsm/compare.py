"""Independent R mirt 1.48 comparison of the local RSM estimator (core.irt_rsm, Rasch identification).

Validation only. Design (fixed before any R run): example datasets irt_027/028/029 (RSM levels 0-2); local
fit with 61 GH nodes; R mirt itemtype "rsm" with slopes 1, latent mean 0 and variance free (reference.R).
Both engines then define theta on the same scale, so no parameter mapping is needed: compared are R's own
probtrace against the local category probabilities on an 81-point grid, the latent SD, each engine's
maximized log-likelihood and the free-parameter count. Criteria (validation/mirt_stage1/criteria.json,
unchanged): probabilities calibration_probability; SD calibration_parameters; log-likelihood common_logLik
(declared now: engine-native values, each on its own fine quadrature). R non-convergence -> NOT VERIFIED.

    python validation/r_mirt_rsm/compare.py --output validation-output/r-mirt-rsm
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CASES = ("irt_027", "irt_028", "irt_029")
THETA = np.linspace(-4, 4, 81)


def local(case):
    sys.path.insert(0, str(ROOT))
    from core import example_datasets as ex
    import core.irt_rsm as rsm
    bundle = ex.materialize(case)
    data = bundle["data"][bundle["items"]]
    return data, rsm.fit(data, points=61), rsm


def compare(case, out, criteria):
    data, f, rsm = local(case)
    folder = out / case; folder.mkdir(parents=True, exist_ok=True)
    data.to_csv(folder / "data.csv", index=False)
    np.savetxt(folder / "theta.csv", THETA, delimiter=",", header="theta", comments="")
    run = subprocess.run(["Rscript", str(ROOT / "validation/r_mirt_rsm/reference.R"), str(folder / "data.csv"),
                          str(folder)], capture_output=True, text=True)
    (folder / "stderr.log").write_text(run.stderr)
    if run.returncode:
        return {"case": case, "status": "FAIL", "reason": "Rscript failed", "stderr": run.stderr[-2000:]}
    r = json.loads((folder / "r_result.json").read_text())
    native = np.exp(rsm.category_log_probs(THETA, f["steps"])).transpose(1, 0, 2)
    report = {"case": case, "N": int(f["n"]), "items": int(f["k"]), "local_converged": bool(f["converged"]),
              "R_converged": bool(r["converged"]), "local_parameters": int(f["npar"]), "R_parameters": int(r["nest"]),
              "local_loglik": f["loglik"], "R_loglik": float(r["loglik"]),
              "calibration_probability": float(np.abs(native - np.asarray(r["probtrace"], float)).max()),
              "calibration_parameters": abs(f["sd"] - float(r["sd"])),
              "common_logLik": abs(f["loglik"] - float(r["loglik"]))}
    checks = {k: report[k] <= criteria[k] for k in ("calibration_probability", "calibration_parameters", "common_logLik")}
    checks["free_parameters"] = report["local_parameters"] == report["R_parameters"]
    report["checks"] = checks
    if not report["R_converged"]:
        report["status"] = "NOT VERIFIED (R did not converge)"
    else:
        report["status"] = ("NUMERICAL AGREEMENT / NOT CERTIFIED" if all(checks.values()) and report["local_converged"]
                            else "FAIL")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if shutil.which("Rscript") is None:
        (out / "report.json").write_text(json.dumps({"status": "NOT VERIFIED", "reason": "Rscript unavailable"}))
        print("NOT VERIFIED: Rscript unavailable"); return 2
    criteria = json.loads((ROOT / "validation/mirt_stage1/criteria.json").read_text())
    reports = [compare(c, out, criteria) for c in CASES]
    (out / "report.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False))
    for r in reports:
        print("R_RSM=" + json.dumps(r, ensure_ascii=False))
    return 0 if all(r["status"].startswith(("NUMERICAL AGREEMENT", "NOT VERIFIED")) for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
