"""Limited deterministic simulation for B10 (NOT a general Type I error or power study).

dif1pl_null : 20 reps, 1PL, R 800 / F 500, focal N(-0.4, 1.2^2), no DIF
dif2pl_null : 10 reps, 2PL, same groups, no DIF
dif2pl_power: 10 reps, 2PL, uniform DIF (M3 b +0.6) and non-uniform DIF (M7 a -0.9)
m2_null     : 20 reps each 1PL / 2PL, k = 10, N = 1000
q3_null     : 5 reps, 2PL, k = 10, N = 800, bootstrap B = 25 with the production estimator
Writes tests/fixtures/irt_b10/simulation_summary.csv. Run: python scripts/irt_b10_simulation.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.special import expit
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.irt import fit_irt
from core.irt_dif import dif_analysis, DIF_TYPES
from core.irt_globalfit import m2_statistic
from core.irt_q3 import q3_analysis

OUT = Path("tests/fixtures/irt_b10")


def two_groups(rng, model, dif):
    k = 12; b = np.linspace(-1.8, 1.8, k)
    a = np.ones(k) if model == "1PL" else np.linspace(.8, 1.8, k)[rng.permutation(k)]
    parts, groups = [], []
    for label, n, mu, sd in (("R", 800, 0., 1.), ("F", 500, -.4, 1.2)):
        aa, bb = a.copy(), b.copy()
        if dif and label == "F":
            bb[2] += .6; aa[6] = max(aa[6] - .9, .25)
        th = rng.normal(mu, sd, n)
        parts.append((rng.random((n, k)) < expit(aa * (th[:, None] - bb))).astype(float)); groups += [label] * n
    return pd.DataFrame(np.vstack(parts), columns=[f"M{i+1}" for i in range(k)]), pd.Series(groups)


def single(rng, model, n=1000, k=10):
    a = np.ones(k) if model == "1PL" else np.linspace(.6, 2., k)[rng.permutation(k)]
    th = rng.normal(0, 1, n); b = np.linspace(-1.6, 1.6, k)
    return pd.DataFrame((rng.random((n, k)) < expit(a * (th[:, None] - b))).astype(float))


def main():
    rows = []
    for study, model, reps, dif, seed in (("dif1pl_null", "1PL", 20, False, 9100), ("dif2pl_null", "2PL", 10, False, 9200),
                                          ("dif2pl_power", "2PL", 10, True, 9300)):
        rng = np.random.default_rng(seed)
        for rep in range(reps):
            x, g = two_groups(rng, model, dif)
            out = dif_analysis(x, g, "R", "F", model=model)
            for _, r in out["table"].iterrows():
                rows.append({"study": study, "rep": rep, "item": r["Madde"], "flag_bh": bool(r["DIF işareti"]),
                             "p_lt_05": bool(r["p"] < .05), "type": r.get("DIF türü", ""), "stable": out["stable"]})
    for model, seed in (("1PL", 9400), ("2PL", 9500)):
        rng = np.random.default_rng(seed)
        for rep in range(20):
            x = single(rng, model); f = fit_irt(x, model=model, points=41, max_iter=500)
            m = m2_statistic(x, f["items"]["a"], f["items"]["b"], model)
            rows.append({"study": f"m2_null_{model}", "rep": rep, "p": m["p"], "M2": m["M2"], "df": m["df"]})
    rng = np.random.default_rng(9600)
    refit = lambda fr: (lambda z: (z["items"]["a"].to_numpy(), z["items"]["b"].to_numpy(), z["converged"]))(
        fit_irt(fr, model="2PL", points=31, max_iter=200))
    for rep in range(5):
        x = single(rng, "2PL", 800); f = fit_irt(x, model="2PL", points=31, max_iter=200)
        q = q3_analysis(x, f["items"]["a"], f["items"]["b"], "2PL", bootstrap=25, seed=rep, points=31, refit=refit)
        rows.append({"study": "q3_null", "rep": rep, "p": q["bootstrap"]["p_max"], "used": q["bootstrap"]["used"]})
    out = pd.DataFrame(rows); out.to_csv(OUT / "simulation_summary.csv", index=False, float_format="%.6g")
    for study in ("dif1pl_null", "dif2pl_null", "dif2pl_power"):
        s = out[out.study == study]; null = s[~s.item.isin(["M3", "M7"])] if study == "dif2pl_power" else s
        print(study, "reps", s.rep.nunique(), "| non-DIF items: p<.05", round(null.p_lt_05.mean(), 4),
              "BH flag", round(null.flag_bh.mean(), 4), "any BH flag/rep", null.groupby("rep").flag_bh.any().sum())
        if study == "dif2pl_power":
            for it in ("M3", "M7"):
                t = s[s.item == it]
                print("   ", it, "detected", int(t.flag_bh.sum()), "/", len(t), "types", t.type.value_counts().to_dict())
    for study in ("m2_null_1PL", "m2_null_2PL", "q3_null"):
        p = out[out.study == study].p
        print(study, "reps", len(p), "p<.05:", int((p < .05).sum()), "median p:", round(p.median(), 3))


if __name__ == "__main__":
    main()
