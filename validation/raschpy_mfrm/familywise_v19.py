"""V19: V18 gates with 1000 replications per cell, null and nonzero (model-specific) scenarios, all 8 models
(PROTOCOL.json V19_recovery_precision_and_nonzero). The generator is written independently of RaschPy.
PYTHONPATH=. python -m validation.raschpy_mfrm.familywise_v19 [--reps 1000] [--workers 4]   -> familywise_v19.json
"""
import argparse, json, warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from validation.raschpy_mfrm.familywise_v18 import evaluate

H = Path(__file__).resolve().parent; SEED = 20261126
MODELS = ("global", "items", "thresholds", "bivector", "matrix", "centrality", "pseudo_halo", "bistretch")
ITEMS = ["K1", "K2", "K3", "K4"]; DELTA = np.array([-.5, -.1, .2, .4]); TAU = np.array([-1.2, 0., 1.2]); K = 3
DESIGNS = {"balanced": (300, ["P1", "P2", "P3"], np.array([-.8, 0., .8]), lambda n, R: [list(range(R))] * n),
           "linked": (400, ["P1", "P2", "P3", "P4", "P5"], np.array([-.8, -.4, 0., .4, .8]),
                      lambda n, R: [[k % 5, (k + 1) % 5] for k in range(n)])}
STRETCH = {"centrality": ("omega",), "pseudo_halo": ("omega",), "bistretch": ("omega_items", "omega_thresholds")}


def centred(R, r, v):
    """Vector over raters: +v for rater r, -v/(R-1) for the others (sums to zero)."""
    out = np.full(R, -v / (R - 1)); out[r] = v; return out


def truth(model, design, scenario):
    """lambda, cells c[r, i, h] (rater term of step h+1), omegas."""
    _, raters, lam, _ = DESIGNS[design]; R = len(raters); I = len(ITEMS)
    c = np.repeat(lam[:, None, None], I, 1).repeat(K, 2).astype(float); om = {}
    if scenario == "null" or model == "global":
        return lam, c, om
    if model in ("items", "bivector"):
        c[:, 2, :] += centred(R, 1, .5)[:, None]                             # second rater harsher on K3
    if model in ("thresholds", "bivector"):
        c += np.outer(centred(R, 0, 1.), np.array([.4, 0., -.4]))[:, None, :]  # lenient rater: compressed steps
    if model == "matrix":
        c[:, 2, 0] += centred(R, 1, .5)                                      # second rater x K3 x first step
    if model in ("centrality", "bistretch"):
        w = np.linspace(.7, 1.3, R) if model == "centrality" else np.linspace(1.3, .7, R)
        c += (w - 1)[:, None, None] * TAU[None, None, :]; om["omega" if model == "centrality" else "omega_thresholds"] = w
    if model in ("pseudo_halo", "bistretch"):
        w = np.linspace(.7, 1.3, R); c += (w - 1)[:, None, None] * DELTA[None, :, None]
        om["omega" if model == "pseudo_halo" else "omega_items"] = w
    return lam, c, om


def simulate(design, c, rng):
    n, raters, _, pairs = DESIGNS[design]; R = len(raters); th = rng.normal(0, 1, n); rows = []
    rr = np.array([(p, r, i) for p, rs in enumerate(pairs(n, R)) for r in rs for i in range(len(ITEMS))])
    b = DELTA[rr[:, 2], None] + TAU[None, :] + c[rr[:, 1], rr[:, 2], :]          # (rows, K)
    num = np.concatenate([np.zeros((len(rr), 1)), np.cumsum(th[rr[:, 0], None] - b, axis=1)], axis=1)
    p = np.exp(num - num.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    x = (rng.random(len(rr))[:, None] > np.cumsum(p, 1)).sum(1)
    return pd.DataFrame({"Person": [f"Z{q:04d}" for q in rr[:, 0]], "Rater": np.array(raters)[rr[:, 1]],
                         "item": np.array(ITEMS)[rr[:, 2]], "score": x})


def fit_params(df, model, design, lam, c, om):
    import raschpy
    raters = DESIGNS[design][1]; ri = {r: k for k, r in enumerate(raters)}
    w = df.pivot(index=["Rater", "Person"], columns="item", values="score").astype(float); w.columns.name = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = raschpy.MFRM(w, max_score=K); m.calibrate(model=model); m.std_errors(model=model, se_method="sandwich")
    out = {}
    it, ise = getattr(m, f"items_{model}"), getattr(m, f"item_se_{model}")
    for k, i in enumerate(ITEMS): out[f"item:{i}"] = (float(it[i]), DELTA[k], float(ise[i]))
    th = np.asarray(getattr(m, f"thresholds_{model}"), float)[-K:]; ths = np.asarray(getattr(m, f"threshold_se_{model}"), float)[-K:]
    for k in range(K): out[f"tau{k + 1}"] = (th[k], TAU[k], ths[k])
    Rv, Se = getattr(m, f"raters_{model}"), getattr(m, f"rater_se_{model}", None)
    if isinstance(Rv, pd.Series):
        for r, v in Rv.items(): out[f"cell:{r}"] = (float(v), lam[ri[r]], float(Se[r]) if isinstance(Se, pd.Series) else np.nan)
    else:
        ok = isinstance(Se, pd.DataFrame) and Se.shape == Rv.shape
        for idx, row in Rv.iterrows():
            r, i = (idx if isinstance(idx, tuple) else (idx, None))
            for pos, (col, v) in enumerate(row.items()):
                if model in ("items", "pseudo_halo"): t = c[ri[r], ITEMS.index(col), 0]
                elif model in ("thresholds", "centrality"): t = c[ri[r], 0, pos]
                else: t = c[ri[r], ITEMS.index(i), pos]
                out[f"cell:{idx}:{col}"] = (float(v), float(t), float(Se.loc[idx, col]) if ok else np.nan)
    for p in STRETCH.get(model, ()):
        est, se = getattr(m, f"{p}_{model}"), getattr(m, f"{p}_se_{model}")
        tw = om.get(p, np.ones(len(raters)))
        for r in est.index: out[f"{p}:{r}"] = (float(est[r]), float(tw[ri[r]]), float(se[r]))
    if model in STRETCH:
        l, ls = getattr(m, f"lambda_{model}"), getattr(m, f"lambda_se_{model}")
        for r in l.index: out[f"lambda:{r}"] = (float(l[r]), lam[ri[r]], float(ls[r]))
    return out


def job(args):
    design, rep = args; rows = []
    for scenario in ("null", "nonzero"):
        rng = np.random.default_rng([SEED, scenario == "nonzero", list(DESIGNS).index(design), rep])
        shared = simulate(design, truth("global", design, "null")[1], rng) if scenario == "null" else None
        for model in MODELS:
            if scenario == "nonzero" and model == "global":
                continue                                                          # global's nonzero = null scenario
            lam, c, om = truth(model, design, scenario)
            d = shared if shared is not None else simulate(design, c, np.random.default_rng([SEED, 1, list(DESIGNS).index(design), rep, MODELS.index(model)]))
            try:
                for k, (e, t, s) in fit_params(d, model, design, lam, c, om).items():
                    rows.append(dict(scenario=scenario, design=design, rep=rep, model=model, key=k, est=e, truth=t, se=s))
            except Exception as ex:
                rows.append(dict(scenario=scenario, design=design, rep=rep, model=model, key="FIT_ERROR", est=np.nan, truth=np.nan,
                                 se=np.nan, err=str(ex)[:80]))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=1000); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    with Pool(a.workers) as p:
        df = pd.DataFrame([r for o in p.imap_unordered(job, [(d, r) for d in DESIGNS for r in range(a.reps)], chunksize=10) for r in o])
    res = {}
    for scen in ("null", "nonzero"):
        sub = df[df.scenario == scen].drop(columns="scenario")
        if scen == "nonzero":                                                    # global: nonzero scenario is the null data
            sub = pd.concat([sub, df[(df.scenario == "null") & (df.model == "global")].drop(columns="scenario")])
        for m, r in evaluate(sub.rename(columns={"design": "design"}).assign(design=sub.design.map({"balanced": "balanced", "linked": "linked"})), a.reps).items():
            res.setdefault(m, {})[scen] = r
    for m, r in res.items():
        r["pass_all"] = all(r[s]["pass_both_designs"] for s in ("null", "nonzero"))
    (H / "familywise_v19.json").write_text(json.dumps(dict(protocol="PROTOCOL.json V19_recovery_precision_and_nonzero", seed=SEED,
                                                            reps=a.reps, results=res), indent=1) + "\n")
    print(json.dumps({m: {s: {d: {k: r[s][d][k] for k in ("max_abs_bias", "se_ratio", "pooled_coverage", "familywise_recovery", "pass")}
                              for d in DESIGNS} for s in ("null", "nonzero")} for m, r in res.items()}, indent=0))
