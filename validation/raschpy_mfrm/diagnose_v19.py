"""V19 exploratory diagnosis (not a gate): linked-design bias of rater-side cells, default vs unregularised RaschPy,
independent re-implementation check. PYTHONPATH=. python -m validation.raschpy_mfrm.diagnose_v19 -> diagnose_v19.json
"""
import json, sys, warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "examples/mfrm")); import build as B
H = Path(__file__).resolve().parent; SEED = 20261125; REPS = 1000
MODELS = ("global", "thresholds", "bivector", "matrix"); R5 = {"P1": -.8, "P2": -.4, "P3": 0., "P4": .4, "P5": .8}
CFG = {"default": {}, "unregularised": dict(constant=0., threshold_constant=0., ridge=0., interaction_ridge=0., stretch_ridge=0.)}


def data(rep):
    r = np.random.default_rng([SEED, rep]); n = 400
    return B._rows(r, [f"D{k:04d}" for k in range(n)], lambda k: [f"P{k % 5 + 1}", f"P{(k + 1) % 5 + 1}"], r.normal(0, 1, n), raters=R5)


def job(rep):
    import raschpy
    d = data(rep); w = d.pivot(index=["rater_id", "person_id"], columns="item_id", values="score").astype(float)
    w.index.names = ["Rater", "Person"]; w.columns.name = None; rows = []
    counts = d.groupby(["rater_id", "score"]).size().unstack(fill_value=0)
    for cfg, kw in CFG.items():
        for model in MODELS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore"); m = raschpy.MFRM(w, max_score=3); m.calibrate(model=model, **kw)
            Rv = getattr(m, f"raters_{model}")
            if isinstance(Rv, pd.Series):
                rows += [dict(rep=rep, cfg=cfg, model=model, key=f"{r}", rater=r, step=None, dev=float(v) - R5[r]) for r, v in Rv.items()]
            else:
                for idx, row in Rv.iterrows():
                    r = idx[0] if isinstance(idx, tuple) else idx
                    rows += [dict(rep=rep, cfg=cfg, model=model, key=f"{idx}:{c}", rater=r, step=int(c), dev=float(v) - R5[r]) for c, v in row.items()]
    return rows, counts.loc["P1"].to_dict()


if __name__ == "__main__":
    with Pool(4) as p:
        res = p.map(job, range(REPS), chunksize=20)
    df = pd.DataFrame([r for rows, _ in res for r in rows]); p1 = pd.DataFrame([c for _, c in res]).mean().to_dict()
    out = {"replications": REPS, "seed": SEED, "P1_mean_category_counts": p1, "models": {}}
    for (cfg, model), g in df.groupby(["cfg", "model"]):
        per = g.groupby("key").dev.agg(["mean", "std", "size"]); per["mcse"] = per["std"] / np.sqrt(per["size"])
        by = g.groupby(["rater", "step"], dropna=False).dev.mean().reset_index()
        out["models"].setdefault(model, {})[cfg] = dict(
            max_abs_bias=float(per["mean"].abs().max()), worst=per["mean"].abs().idxmax(), worst_bias=float(per["mean"][per["mean"].abs().idxmax()]),
            worst_mcse=float(per.loc[per["mean"].abs().idxmax(), "mcse"]), cells_over_005=int((per["mean"].abs() > .05).sum()), n_cells=len(per),
            bias_by_rater_step={f"{r}:{s}": round(float(b), 4) for r, s, b in by.itertuples(index=False)})
    # independent estimator (validation/raschpy_mfrm/independent.py) on 10 replications, default regularisation
    import validation.raschpy_mfrm.independent as V
    d8 = []
    for rep in range(10):
        dd = data(rep).rename(columns={"person_id": "person", "item_id": "item", "rater_id": "rater"})
        d8 += [V.compare(dd, m, 3, True)[0] for m in ("thresholds", "bivector", "matrix")]
    out["independent_estimator_max_grid_diff"] = float(max(d8))
    (H / "diagnose_v19.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({m: {c: {k: v[k] for k in ("max_abs_bias", "worst", "worst_bias", "worst_mcse", "cells_over_005")} for c, v in x.items()}
                      for m, x in out["models"].items()}, indent=1), out["independent_estimator_max_grid_diff"], p1)
