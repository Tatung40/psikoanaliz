"""Limited deterministic simulation for B11 (NOT a general Type I error or power study).

dif3pl_null : 8 reps, 3PL, R 1500 / F 1000, focal N(-0.3, 1.1^2), no DIF, GH 31 points.
              Records, per item, whether the omnibus LR got a p-value (interior) and p < .05,
              and the LR statistic of items whose p-value is withheld (boundary).
m2_null_3pl : 10 reps, 3PL, k = 12, N = 2000 (production estimator EM + L-BFGS-B):
              share of fits with a reportable (interior) M2 and p < .05 among them.
Writes tests/fixtures/irt_b11/simulation_<study>.csv. Run: python scripts/irt_b11_simulation.py [study ...]
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.special import expit
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.irt_dif34 import dif_analysis_34
from core.irt_globalfit import global_fit_34
from ui.irt_optimization import calibration_settings, run_irt_calibration

OUT = Path("tests/fixtures/irt_b11")


def items(rng, k, model):
    a = np.linspace(.9, 2.0, k)[rng.permutation(k)]; b = np.linspace(-.4, 2.2, k)
    c = np.linspace(.15, .30, k)[rng.permutation(k)]
    d = np.linspace(.88, .98, k)[rng.permutation(k)] if model == "4PL" else np.ones(k)
    return a, b, c, d


def sim(rng, n, mu, sd, a, b, c, d):
    th = rng.normal(mu, sd, n)
    return (rng.random((n, len(a))) < c + (d - c) * expit(a * (th[:, None] - b))).astype(float)


def dif_null(model="3PL", reps=8, seed=11000):
    rng = np.random.default_rng(seed); rows = []
    for rep in range(reps):
        a, b, c, d = items(rng, 12, model)
        x = pd.DataFrame(np.vstack([sim(rng, 1500, 0, 1, a, b, c, d), sim(rng, 1000, -.3, 1.1, a, b, c, d)]),
                         columns=[f"M{i+1}" for i in range(12)])
        g = pd.Series(["R"] * 1500 + ["F"] * 1000)
        out = dif_analysis_34(x, g, "R", "F", model=model, points=31, purify=False)
        for _, r in out["table"].iterrows():
            rows.append({"rep": rep, "item": r["Madde"], "has_p": bool(np.isfinite(r["p"])), "p": r["p"],
                         "lr": r[[c_ for c_ in out["table"].columns if c_.startswith("LR χ² genel")][0]]})
        print("rep", rep, "done", flush=True)
    out = pd.DataFrame(rows); out.to_csv(OUT / f"simulation_dif_{model.lower()}_null.csv", index=False, float_format="%.6g")
    ok = out[out.has_p]
    print(model, "dif null: item-tests", len(out), "with p", len(ok), "p<.05", int((ok.p < .05).sum()),
          "| boundary items LR > chi2(.95)", int((out[~out.has_p].lr > 7.815).sum()), "of", int((~out.has_p).sum()))


def m2_null(model="3PL", reps=10, seed=12000):
    rng = np.random.default_rng(seed); rows = []
    for rep in range(reps):
        a, b, c, d = items(rng, 12, model)
        x = pd.DataFrame(sim(rng, 2000, 0, 1, a, b, c, d))
        f = run_irt_calibration(x, calibration_settings(model, 31, 500, refine=True))
        from core.irt_se_diagnostics import hessian_boundary_diagnostics
        diag = hessian_boundary_diagnostics(f["items"].assign(Madde=np.arange(12)), model)
        fixed = {p: np.isin(np.arange(12), diag.loc[diag["Parametre"] == p, "Madde"].astype(int)) for p in "abcd"}
        g = global_fit_34(x, f["items"]["a"], f["items"]["b"], f["items"]["c"], f["items"]["d"], model,
                          np.zeros(len(x)), "EAP", f["converged"], fixed, points=31)
        m = g["m2"]
        rows.append({"rep": rep, "available": m["available"], "conditional": m.get("conditional", np.nan),
                     "M2": m.get("M2", np.nan), "df": m.get("df", np.nan), "p": m.get("p", np.nan)})
    out = pd.DataFrame(rows); out.to_csv(OUT / f"simulation_m2_{model.lower()}_null.csv", index=False, float_format="%.6g")
    reg = out[out.available & (out.conditional == False)]                          # noqa: E712
    print(model, "m2 null: reps", reps, "interior (inferential)", len(reg), "p<.05", int((reg.p < .05).sum()),
          "| conditional", int((out.conditional == True).sum()), "p<.05 among them", int((out[out.conditional == True].p < .05).sum()))


if __name__ == "__main__":
    todo = sys.argv[1:] or ["dif3pl", "m2_3pl"]
    if "dif3pl" in todo: dif_null("3PL")
    if "m2_3pl" in todo: m2_null("3PL")
    if "dif4pl" in todo: dif_null("4PL", reps=5, seed=13000)
    if "m2_4pl" in todo: m2_null("4PL", seed=14000)
