"""V12: Wald 95% CI coverage of RaschPy sandwich SE (rater-side parameters, items, thresholds) for all 8 models
(PROTOCOL.json amendment_2026_11_22). Reuses the V9 null generator.
PYTHONPATH=. python -m validation.raschpy_mfrm.simulation_v12 [--reps 100] [--workers 4]   -> simulation_v12.json
"""
import argparse, json
from multiprocessing import Pool

import numpy as np
import pandas as pd

from validation.raschpy_mfrm.simulation import DESIGNS, H, ITEMS, MODELS, SEED, TAU, cells, fit, sandwich, simulate, truth

Z = 1.959964
GATE = dict(pooled=(0.92, 0.98), min_param=0.85, ratio=(0.80, 1.25))


def job(args):
    design, N, rep = args
    rng = np.random.default_rng([SEED, 12, N, rep, list(DESIGNS).index(design)]); d = simulate(design, N, rng); out = []
    for model in MODELS:
        try:
            m = fit(d, model)
        except Exception as e:                                   # convergence/numerical failure: counted, never dropped silently
            out.append(dict(design=design, N=N, rep=rep, model=model, group="FIT_ERROR", key=str(e)[:60])); continue
        se = sandwich(m, model); est = cells(m, model)
        for k, v in est.items():
            if k in se:
                out.append(dict(design=design, N=N, rep=rep, model=model, group="rater", key=k, value=v, truth=truth(k, design), se=se[k]))
        it, its = getattr(m, f"items_{model}"), getattr(m, f"item_se_{model}")
        for i, t in ITEMS.items():
            out.append(dict(design=design, N=N, rep=rep, model=model, group="items", key=f"item:{i}", value=float(it[i]), truth=t,
                            se=float(its[i])))
        th, ths = np.asarray(getattr(m, f"thresholds_{model}"), float)[-3:], np.asarray(getattr(m, f"threshold_se_{model}"), float)[-3:]
        for k in range(3):
            out.append(dict(design=design, N=N, rep=rep, model=model, group="thresholds", key=f"tau{k + 1}", value=th[k],
                            truth=TAU[k], se=ths[k]))
    return out


def summarise(df, reps):
    ok = df[df.group != "FIT_ERROR"].copy()
    ok["covered"] = (ok.value - ok.truth).abs() <= Z * ok.se; ok["dev"] = ok.value - ok.truth
    res = {}
    for (model, group, design, N), g in ok.groupby(["model", "group", "design", "N"]):
        per = g.groupby("key").agg(cov=("covered", "mean"), se=("se", "mean"), sd=("dev", "std"), bias=("dev", "mean"),
                                   finite=("se", lambda s: bool(np.isfinite(s).all())))
        ratio = per.se / per.sd; pooled = float(g.covered.mean())
        passed = bool(GATE["pooled"][0] <= pooled <= GATE["pooled"][1] and per["cov"].min() >= GATE["min_param"]
                      and ratio.between(*GATE["ratio"]).all() and per.finite.all())
        res.setdefault(model, {}).setdefault(group, {})[f"{design}:N{N}"] = dict(
            pooled_coverage=pooled, min_param_coverage=float(per["cov"].min()), se_over_sd=[float(ratio.min()), float(ratio.max())],
            max_abs_bias=float(per.bias.abs().max()), max_rmse=float(np.sqrt(per.bias ** 2 + per.sd ** 2).max()), n_params=len(per),
            gate=passed)
    for model, groups in res.items():
        fails = int(((df.model == model) & (df.group == "FIT_ERROR")).sum())
        for group, cellsd in groups.items():
            cellsd["all_cells_pass"] = all(v["gate"] for v in cellsd.values())
        groups["convergence"] = dict(fit_failures=fails, fits=int(df[df.model == model][["design", "N", "rep"]].drop_duplicates().shape[0]))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=100); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    jobs = [(dsg, N, r) for dsg in DESIGNS for N in (250, 1000) for r in range(a.reps)]
    with Pool(a.workers) as pool:
        df = pd.DataFrame([row for out in pool.imap_unordered(job, jobs, chunksize=4) for row in out])
    res = summarise(df, a.reps)
    (H / "simulation_v12.json").write_text(json.dumps(dict(protocol="PROTOCOL.json amendment_2026_11_22 V12", seed=SEED, reps=a.reps,
                                                            gate=GATE, results=res), indent=1) + "\n")
    print(json.dumps({m: {g: v.get("all_cells_pass") for g, v in r.items() if g != "convergence"} | {"fails": r["convergence"]["fit_failures"]}
                      for m, r in res.items()}, indent=0))
