"""V14: person CSEM coverage (all 8 models) and V14b stretch-model lambda SE coverage (PROTOCOL amendment_2026_11_22b).
PYTHONPATH=. python -m validation.raschpy_mfrm.simulation_v14 [--workers 2]   -> simulation_v14.json
"""
import argparse, json
from multiprocessing import Pool

import numpy as np
import pandas as pd

from validation.raschpy_mfrm.simulation import DESIGNS, H, MODELS, SEED, fit, simulate

Z = 1.959964; STRETCH = ("centrality", "pseudo_halo", "bistretch"); REPS = {m: (100 if m in STRETCH else 20) for m in MODELS}
GATE = dict(csem=(0.92, 0.98), pooled=(0.92, 0.98), min_param=0.85, ratio=(0.80, 1.25))


def job(args):
    design, N, rep = args
    d, th = simulate(design, N, np.random.default_rng([SEED, 14, N, rep, list(DESIGNS).index(design)]), return_theta=True)
    out = []
    for model in [m for m in MODELS if rep < REPS[m]]:
        try:
            m = fit(d, model); m.person_stats_df(model=model)
        except Exception as e:
            out.append(dict(kind="FIT_ERROR", model=model, design=design, N=N, rep=rep, msg=str(e)[:60])); continue
        p = m.person_stats_df(model=model); ne = (p.Score > 0) & (p.Score < p["Max score"])
        est, csem = getattr(m, f"persons_{model}"), getattr(m, f"csem_vector_{model}")
        idx = p.index[ne]; cov = (np.abs(est[idx] - th[idx]) <= Z * csem[idx])
        out.append(dict(kind="csem", model=model, design=design, N=N, rep=rep, covered=int(cov.sum()), n=int(len(idx)),
                        extreme=int((~ne).sum()), se_mean=float(csem[idx].mean()), err_sd=float((est[idx] - th[idx]).std())))
        if model in STRETCH:
            lam, se = getattr(m, f"lambda_{model}"), getattr(m, f"lambda_se_{model}")
            for r, v in lam.items():
                out.append(dict(kind="lambda", model=model, design=design, N=N, rep=rep, key=r, value=float(v),
                                truth=DESIGNS[design][r], se=float(se[r])))
    return out


def summarise(df):
    res = {}
    c = df[df.kind == "csem"]
    for (model, design, N), g in c.groupby(["model", "design", "N"]):
        cov = g.covered.sum() / g.n.sum()
        res.setdefault(model, {}).setdefault("csem", {})[f"{design}:N{N}"] = dict(
            coverage=float(cov), persons=int(g.n.sum()), extreme_share=float(g.extreme.sum() / (g.extreme.sum() + g.n.sum())),
            mean_csem=float(g.se_mean.mean()), mean_error_sd=float(g.err_sd.mean()), gate=bool(GATE["csem"][0] <= cov <= GATE["csem"][1]))
    lam = df[df.kind == "lambda"].copy()
    lam["covered"] = (lam.value - lam.truth).abs() <= Z * lam.se; lam["dev"] = lam.value - lam.truth
    for (model, design, N), g in lam.groupby(["model", "design", "N"]):
        per = g.groupby("key").agg(cov=("covered", "mean"), se=("se", "mean"), sd=("dev", "std"), bias=("dev", "mean"))
        ratio = per.se / per.sd; pooled = float(g.covered.mean())
        res[model].setdefault("lambda", {})[f"{design}:N{N}"] = dict(
            pooled_coverage=pooled, min_param_coverage=float(per["cov"].min()), se_over_sd=[float(ratio.min()), float(ratio.max())],
            max_abs_bias=float(per.bias.abs().max()), gate=bool(GATE["pooled"][0] <= pooled <= GATE["pooled"][1]
                                                              and per["cov"].min() >= GATE["min_param"] and ratio.between(*GATE["ratio"]).all()))
    for model, r in res.items():
        for k in ("csem", "lambda"):
            if k in r: r[k]["all_cells_pass"] = all(v["gate"] for v in r[k].values())
        r["fit_failures"] = int(((df.kind == "FIT_ERROR") & (df.model == model)).sum())
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=2); a = ap.parse_args()
    jobs = [(dsg, N, r) for dsg in DESIGNS for N in (250, 1000) for r in range(max(REPS.values()))]
    with Pool(a.workers) as pool:
        df = pd.DataFrame([row for out in pool.imap_unordered(job, jobs, chunksize=4) for row in out])
    res = summarise(df)
    (H / "simulation_v14.json").write_text(json.dumps(dict(protocol="PROTOCOL.json amendment_2026_11_22b V14", seed=SEED, reps=REPS,
                                                            gate=GATE, results=res), indent=1) + "\n")
    print(json.dumps({m: {k: r[k]["all_cells_pass"] for k in ("csem", "lambda") if k in r} for m, r in res.items()}))
