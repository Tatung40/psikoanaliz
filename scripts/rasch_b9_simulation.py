"""Limited deterministic simulation for B9 (NOT a general Type I error study).

dif_null   : 25 replications, k=12, reference N=600, focal N=300, focal mean -0.5, SD 1.2,
             no DIF; BH alpha=.05 with purification.
dif_power  : same design with uniform DIF +0.6 (M3) and -0.5 (M8).
m2_null    : 25 replications, Rasch-consistent, k=10, N=800, sigma=1.1.
q3_null    : 10 replications, k=10, N=600, bootstrap B=50 per replication.
Writes tests/fixtures/rasch_b9/simulation_summary.csv. Run: python scripts/rasch_b9_simulation.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.special import expit
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.rasch import fit_rasch
from core.rasch_dif import dif_analysis
from core.rasch_globalfit import m2_statistic
from core.rasch_q3 import q3_analysis

OUT = Path("tests/fixtures/rasch_b9")


def two_groups(rng, dif):
    k = 12; b = np.linspace(-1.8, 1.8, k); shift = np.zeros(k)
    if dif:
        shift[2], shift[7] = .6, -.5
    parts, groups = [], []
    for label, n, mu, sd, s in (("R", 600, 0., 1., 0.), ("F", 300, -.5, 1.2, 1.)):
        th = rng.normal(mu, sd, n)
        parts.append((rng.random((n, k)) < expit(th[:, None] - (b + s * shift))).astype(float)); groups += [label] * n
    return pd.DataFrame(np.vstack(parts), columns=[f"M{i+1}" for i in range(k)]), pd.Series(groups)


def rasch_data(rng, n, k, sd):
    th = rng.normal(0, sd, n); b = np.linspace(-1.6, 1.6, k)
    return pd.DataFrame((rng.random((n, k)) < expit(th[:, None] - b)).astype(float))


def main():
    rows = []
    for design in ("dif_null", "dif_power"):
        rng = np.random.default_rng(5000 if design == "dif_null" else 6000)
        for rep in range(25):
            x, g = two_groups(rng, design == "dif_power")
            out = dif_analysis(x, g, "R", "F")
            t = out["table"]
            for _, r in t.iterrows():
                rows.append({"study": design, "rep": rep, "item": r["Madde"], "flag_bh": bool(r["DIF işareti"]),
                             "p_lt_05": bool(r["p"] < .05), "delta": r["Δ (b_odak − b_ref)"], "stable": out["stable"]})
    rng = np.random.default_rng(7000)
    for rep in range(25):
        x = rasch_data(rng, 800, 10, 1.1)
        f = fit_rasch(x)
        m = m2_statistic(f["data"], f["b"], f["sd"])
        rows.append({"study": "m2_null", "rep": rep, "item": "", "p": m["p"], "M2": m["M2"], "df": m["df"]})
    rng = np.random.default_rng(8000)
    for rep in range(10):
        x = rasch_data(rng, 600, 10, 1.)
        q = q3_analysis(x, bootstrap=50, seed=100 + rep)
        rows.append({"study": "q3_null", "rep": rep, "item": "", "p": q["bootstrap"]["p_max"]})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "simulation_summary.csv", index=False, float_format="%.6g")
    d = out[out.study.str.startswith("dif")]
    dif_items = {"M3", "M8"}
    for study in ("dif_null", "dif_power"):
        s = d[d.study == study]
        null_items = s[~s.item.isin(dif_items)] if study == "dif_power" else s
        print(study, "reps", s.rep.nunique(), "| any BH flag on non-DIF items:",
              null_items.groupby("rep").flag_bh.any().mean(),
              "| per-test BH flag rate (non-DIF):", round(null_items.flag_bh.mean(), 4),
              "| per-test p<.05 (non-DIF):", round(null_items.p_lt_05.mean(), 4))
        if study == "dif_power":
            print("   detection M3/M8:", s[s.item == "M3"].flag_bh.mean(), s[s.item == "M8"].flag_bh.mean())
    for study in ("m2_null", "q3_null"):
        p = out[out.study == study].p
        print(study, "reps", len(p), "share p<.05:", round((p < .05).mean(), 3), "median p:", round(p.median(), 3))


if __name__ == "__main__":
    main()
