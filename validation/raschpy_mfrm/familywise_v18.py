"""V18: prospective family-wise parameter recovery for all 8 RaschPy models (PROTOCOL.json V18_familywise_recovery).
PYTHONPATH=. python -m validation.raschpy_mfrm.familywise_v18 [--reps 200] [--workers 4]   -> familywise_v18.json
"""
import argparse, json, sys, warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "examples/mfrm"))
import build as B                                                         # the fixture generator

H = Path(__file__).resolve().parent; SEED = 20261124
MODELS = ("global", "items", "thresholds", "bivector", "matrix", "centrality", "pseudo_halo", "bistretch")
STRETCH = {"centrality": ("omega",), "pseudo_halo": ("omega",), "bistretch": ("omega_items", "omega_thresholds")}
R5 = {"P1": -.8, "P2": -.4, "P3": 0., "P4": .4, "P5": .8}
DESIGN = {"balanced": (300, lambda k: list(B.RATERS), dict(B.RATERS)),
          "linked": (400, lambda k: [f"P{k % 5 + 1}", f"P{(k + 1) % 5 + 1}"], R5)}


def data(design, rep):
    n, pairs, raters = DESIGN[design]; r = np.random.default_rng([SEED, list(DESIGN).index(design), rep])
    return B._rows(r, [f"V{k:04d}" for k in range(n)], pairs, r.normal(0, 1, n), raters=raters)


def params(m, model, raters):
    """{key: (estimate, truth, se or nan)} for items, thresholds, rater-side cells, stretch omegas and lambdas."""
    out = {}
    it, ise = getattr(m, f"items_{model}"), getattr(m, f"item_se_{model}")
    for i, t in B.ITEMS.items(): out[f"item:{i}"] = (float(it[i]), t, float(ise[i]))
    th = np.asarray(getattr(m, f"thresholds_{model}"), float)[-3:]; ths = np.asarray(getattr(m, f"threshold_se_{model}"), float)[-3:]
    for k in range(3): out[f"tau{k + 1}"] = (th[k], float(B.TAU[k]), ths[k])
    Rv, Se = getattr(m, f"raters_{model}"), getattr(m, f"rater_se_{model}", None)
    if isinstance(Rv, pd.Series):
        for r, v in Rv.items(): out[f"cell:{r}"] = (float(v), raters[r], float(Se[r]) if isinstance(Se, pd.Series) else np.nan)
    else:
        ok = isinstance(Se, pd.DataFrame) and Se.shape == Rv.shape
        for idx, row in Rv.iterrows():
            r = idx[0] if isinstance(idx, tuple) else idx
            for c, v in row.items(): out[f"cell:{idx}:{c}"] = (float(v), raters[r], float(Se.loc[idx, c]) if ok else np.nan)
    for p in STRETCH.get(model, ()):
        est, se = getattr(m, f"{p}_{model}"), getattr(m, f"{p}_se_{model}")
        for r in est.index: out[f"{p}:{r}"] = (float(est[r]), 1.0, float(se[r]))
    if model in STRETCH:
        lam, lse = getattr(m, f"lambda_{model}"), getattr(m, f"lambda_se_{model}")
        for r in lam.index: out[f"lambda:{r}"] = (float(lam[r]), raters[r], float(lse[r]))
    return out


def job(args):
    import raschpy
    design, rep = args; d = data(design, rep); raters = DESIGN[design][2]
    w = d.pivot(index=["rater_id", "person_id"], columns="item_id", values="score").astype(float)
    w.index.names = ["Rater", "Person"]; w.columns.name = None; rows = []
    for model in MODELS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = raschpy.MFRM(w, max_score=3); m.calibrate(model=model); m.std_errors(model=model, se_method="sandwich")
            for k, (e, t, s) in params(m, model, raters).items():
                rows.append(dict(design=design, rep=rep, model=model, key=k, est=e, truth=t, se=s))
        except Exception as e:
            rows.append(dict(design=design, rep=rep, model=model, key="FIT_ERROR", est=np.nan, truth=np.nan, se=np.nan, err=str(e)[:80]))
    return rows


def evaluate(df, reps):
    half = reps // 2; out = {}
    for (model, design), g in df.groupby(["model", "design"]):
        fails = int((g.key == "FIT_ERROR").sum()); g = g[g.key != "FIT_ERROR"].assign(dev=lambda x: x.est - x.truth)
        cal, con = g[g.rep < half], g[g.rep >= half]
        sd_cal = cal.groupby("key").dev.std(); per = con.groupby("key").agg(bias=("dev", "mean"), sd=("dev", "std"), n=("dev", "size"),
                                                                             se=("se", "mean"))
        bias_ok = bool((per.bias.abs() <= np.maximum(3 * per.sd / np.sqrt(per.n), .05)).all())
        withse = per[per.se.notna()]; ratio = (withse.se / withse.sd)
        con_se = con[con.se.notna()].assign(cov=lambda x: (x.dev.abs() <= 1.959964 * x.se))
        cov_par = con_se.groupby("key")["cov"].mean(); pooled = float(con_se["cov"].mean()) if len(con_se) else None
        K = per.shape[0]; crit = float(norm.ppf(1 - .025 / K))
        z = con.assign(z=lambda x: x.dev.abs() / x.key.map(sd_cal)).groupby("rep").z.max()
        fw = float((z <= crit).mean())
        res = dict(fit_failures=fails, n_params=K, n_with_se=int(len(withse)), confirmatory_replications=int(z.shape[0]),
                   max_abs_bias=float(per.bias.abs().max()), se_ratio=[float(ratio.min()), float(ratio.max())] if len(ratio) else None,
                   pooled_coverage=pooled, min_param_coverage=float(cov_par.min()) if len(cov_par) else None,
                   bonferroni_critical_z=crit, familywise_recovery=fw,
                   gate_bias=bias_ok, gate_se_ratio=bool(len(ratio) and ratio.between(.80, 1.25).all()),
                   gate_coverage=bool(pooled is not None and .92 <= pooled <= .98 and cov_par.min() >= .85),
                   gate_familywise=bool(fw >= .90))
        res["pass"] = bool(fails == 0 and res["gate_bias"] and res["gate_se_ratio"] and res["gate_coverage"] and res["gate_familywise"])
        out.setdefault(model, {})[design] = res
    for model, r in out.items():
        r["pass_both_designs"] = all(r[d]["pass"] for d in DESIGN)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=200); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    with Pool(a.workers) as p:
        df = pd.DataFrame([row for o in p.imap_unordered(job, [(d, r) for d in DESIGN for r in range(a.reps)], chunksize=4) for row in o])
    res = evaluate(df, a.reps)
    (H / "familywise_v18.json").write_text(json.dumps(dict(protocol="PROTOCOL.json V18_familywise_recovery", seed=SEED,
                                                            reps=a.reps, results=res), indent=1) + "\n")
    print(json.dumps({m: {d: {k: r[d][k] for k in ("max_abs_bias", "se_ratio", "pooled_coverage", "familywise_recovery", "pass")}
                          for d in DESIGN} for m, r in res.items()}, indent=1))
