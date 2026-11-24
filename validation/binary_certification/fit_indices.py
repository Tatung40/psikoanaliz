"""Model-fit outputs (all binary models): LogLik vs an independent marginal likelihood (61 GH nodes, same as the fit),
parameter count, -2LL, AIC = -2LL + 2k, BIC = -2LL + k ln N; MAP flagged (LogLik is the likelihood at the MAP point).

    PYTHONPATH=. python validation/binary_certification/fit_indices.py --output validation/evidence/binary_certification/fit_indices.json
"""
import argparse, json
from pathlib import Path
import numpy as np
from validation.binary_certification.common import Objective, simulate

CASES = [("Rasch", None, "ML", 1000, 12), ("1PL", "common_a", "ML", 1000, 12), ("2PL", None, "ML", 1000, 12),
         ("3PL", None, "MAP", 3000, 12), ("4PL", None, "MAP", 3000, 12), ("3PL", None, "ML", 3000, 12)]


def check(model, variant, est, n, k):
    from core.irt_library_diagnostics import library_fit_diagnostics
    from core.irt_python_mirt_adapter import fit_python_mirt
    gen = "1PL" if model == "Rasch" else model
    x = simulate(gen, n, k, 31, variant)
    kw = dict(tol=1e-10, max_iter=50000) if model == "4PL" else {}
    fit = fit_python_mirt(x, model=model, estimation=est, n_quadpts=61, common_discrimination=variant == "common_a", **kw).fit
    d = library_fit_diagnostics(fit); co = fit.coef()
    p = {"a": co["discrimination"].to_numpy(float) if "discrimination" in co else np.ones(k), "b": co["difficulty"].to_numpy(float)}
    for q, c in (("c", "guessing"), ("d", "upper")):
        if c in co: p[q] = co[c].to_numpy(float)
    ll = Objective(x, gen, map_priors=False, points=61).value(p)
    npar = {"Rasch": k, "1PL": k + 1, "2PL": 2 * k, "3PL": 3 * k, "4PL": 4 * k}[model]
    m2 = -2 * d["log_likelihood"]
    checks = {"converged": bool(d["converged"]), "loglik_independent": abs(d["log_likelihood"] - ll) <= 1e-6,
              "n_parameters": d["n_parameters"] == npar, "n_observations": d["n_observations"] == n,
              "aic": abs(d["aic"] - (m2 + 2 * npar)) <= 1e-8, "bic": abs(d["bic"] - (m2 + npar * np.log(n))) <= 1e-8}
    return {"model": model, "variant": variant, "estimation": est, "N": n, "k": k, "loglik": d["log_likelihood"],
            "loglik_independent": ll, "n_parameters": d["n_parameters"], "expected_n_parameters": npar,
            "aic": d["aic"], "bic": d["bic"], "checks": checks, "pass": all(checks.values())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--output", required=True); a = ap.parse_args()
    rows = [check(*c) for c in CASES]
    Path(a.output).write_text(json.dumps({"cases": rows, "all_pass": all(r["pass"] for r in rows)}, indent=1, default=float) + "\n")
    for r in rows: print(r["model"], r["variant"], r["estimation"], r["pass"], {k: v for k, v in r["checks"].items() if not v}, f"dLL {r['loglik'] - r['loglik_independent']:.1e}", r["n_parameters"], r["expected_n_parameters"])
