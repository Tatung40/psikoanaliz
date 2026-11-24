"""Amendment D3: in-scope Rasch sample (N = 2000, k = 20) generated and independently validated.

    PYTHONPATH=. python validation/binary_certification/d3_rasch_sample.py --output validation/evidence/binary_certification/d3_rasch_sample.json
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "tests/fixtures/rasch/data_rasch_inscope.csv"
B = np.linspace(-2, 2, 20)


def generate():
    rng = np.random.default_rng(20261124)
    th = rng.standard_normal(2000)
    x = (rng.random((2000, 20)) < expit(th[:, None] - B)).astype(int)
    pd.DataFrame(x, columns=[f"M{j + 1}" for j in range(20)]).to_csv(SAMPLE, index=False)


def validate():
    from core.irt_binary_inference import parameter_inference
    from core.irt_binary_usage_policy import certification
    from core.irt_python_mirt_adapter import fit_python_mirt
    from validation.binary_certification.common import Objective
    from validation.binary_certification.reference_numpy import fit
    x = pd.read_csv(SAMPLE).to_numpy(int); k = x.shape[1]
    out = fit_python_mirt(x, model="Rasch", n_quadpts=61); b = out.fit.coef().difficulty.to_numpy(float)
    ref, info = fit(x, "1PL"); obj = Objective(x, "1PL")
    f_py, f_ref = obj.value({"a": np.ones(k), "b": b}), obj.value({"a": np.ones(k), "b": ref["b"]})
    table, _ = parameter_inference(out.fit, "Rasch", [f"M{j + 1}" for j in range(k)]); se = table[table.Parametre == "b"].SH.to_numpy()
    z = np.abs(b - B) / se
    state, _ = certification("Rasch", out.estimation, x.shape[0], k, out.solver_settings)
    checks = {"converged": bool(out.fit.converged) and info["converged"], "b_vs_reference": float(np.abs(b - ref["b"]).max()) <= .002,
              "objective": abs(f_py - f_ref) <= .001, "recovery_3se": bool(np.all(z <= 3)), "certified": state == "certified"}
    return {"sample": str(SAMPLE.relative_to(ROOT)), "sha256": hashlib.sha256(SAMPLE.read_bytes()).hexdigest(), "shape": list(x.shape),
            "max_abs_diff_b_reference": float(np.abs(b - ref["b"]).max()), "objective_gap": f_py - f_ref,
            "max_abs_z_vs_truth": float(z.max()), "certification_state": state, "checks": checks, "pass": all(checks.values())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--output", required=True); a = ap.parse_args()
    if not SAMPLE.exists():
        generate()
    res = validate(); Path(a.output).write_text(json.dumps(res, indent=1) + "\n"); print(json.dumps(res, indent=1))
