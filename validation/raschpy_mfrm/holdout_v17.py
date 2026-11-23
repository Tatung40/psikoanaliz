"""V17: independent holdout replications of the fixture designs (PROTOCOL amendment_2026_11_23b; diagnostic).
PYTHONPATH=. python -m validation.raschpy_mfrm.holdout_v17 [--reps 200] [--workers 2]   -> holdout_v17.json
"""
import argparse, json, sys, warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "examples/mfrm"))
import build as B                                                         # the fixture generator itself

H = Path(__file__).resolve().parent; MODELS = ("bivector", "centrality", "bistretch", "matrix"); SEED = 20261123
TRUTH = json.loads((ROOT / "examples/mfrm/fixtures.json").read_text())["fixtures"]
DESIGN = {"balanced_3facet": (300, lambda k: list(B.RATERS)), "unbalanced_linked": (400, lambda k: [f"P{k % 5 + 1}", f"P{(k + 1) % 5 + 1}"])}


def data(fx, rep):
    n, pairs = DESIGN[fx]; r = np.random.default_rng([SEED, rep, len(fx)]); ids = [f"H{k:04d}" for k in range(n)]
    th = r.normal(0, 1, n); raters = TRUTH[fx]["truth"]["raters"]
    return B._rows(r, ids, pairs, th, raters=raters) if fx == "unbalanced_linked" else B._rows(r, ids, pairs, th)


def cells(df, model):
    import raschpy
    w = df.pivot(index=["rater_id", "person_id"], columns="item_id", values="score").astype(float)
    w.index.names = ["Rater", "Person"]; w.columns.name = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); m = raschpy.MFRM(w, max_score=3); m.calibrate(model=model)
    Rv = getattr(m, f"raters_{model}"); out = {}
    for idx, row in Rv.iterrows():
        r = idx[0] if isinstance(idx, tuple) else idx
        for c, v in row.items(): out[f"{idx}:{c}"] = (float(v), r)
    return out


def job(args):
    fx, rep = args; d = data(fx, rep) if rep >= 0 else pd.read_csv(ROOT / f"examples/mfrm/{fx}.csv"); t = TRUTH[fx]["truth"]["raters"]; out = []
    for model in MODELS:
        try:
            for k, (v, r) in cells(d, model).items():
                out.append(dict(fixture=fx, rep=rep, model=model, key=k, dev=v - t[r]))
        except Exception as e:
            out.append(dict(fixture=fx, rep=rep, model=model, key="FIT_ERROR", dev=np.nan))
    return out


def summarise(df):
    res = {}; hold = df[(df.rep >= 0) & (df.key != "FIT_ERROR")]; fixd = df[df.rep < 0]
    for (fx, model), g in hold.groupby(["fixture", "model"]):
        per = g.groupby("key").dev.agg(["mean", "std", "size"]); mcse = per["std"] / np.sqrt(per["size"])
        tol = np.maximum(3 * per["std"], .30)
        wide = g.pivot(index="rep", columns="key", values="dev")
        fail_rate = float((wide.abs() > tol[wide.columns]).any(axis=1).mean())
        zmax = (wide.abs() / per["std"][wide.columns]).max(axis=1)
        f = fixd[(fixd.fixture == fx) & (fixd.model == model)].set_index("key").dev
        fz = float((f.abs() / per["std"].reindex(f.index)).max())
        worst = per["mean"].abs().idxmax()
        res[f"{model}:{fx}"] = dict(replications=int(wide.shape[0]), fit_failures=int(((df.key == "FIT_ERROR") & (df.model == model) & (df.fixture == fx)).sum()),
            max_abs_bias=float(per["mean"].abs().max()), worst_cell=worst, worst_cell_bias=float(per["mean"][worst]),
            max_sd=float(per["std"].max()), max_rmse=float(np.sqrt(per["mean"] ** 2 + per["std"] ** 2).max()),
            bias_gate=bool((per["mean"].abs() <= np.maximum(3 * mcse, .05)).all()),
            p_replicate_fails_v5=fail_rate, fixture_max_z=fz, fixture_tail_prob=float((zmax >= fz).mean()))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=200); ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(); jobs = [(fx, r) for fx in DESIGN for r in [-1] + list(range(a.reps))]
    with Pool(a.workers) as p:
        df = pd.DataFrame([row for out in p.imap_unordered(job, jobs, chunksize=8) for row in out])
    res = summarise(df)
    (H / "holdout_v17.json").write_text(json.dumps(dict(protocol="PROTOCOL.json amendment_2026_11_23b V17", seed=SEED, reps=a.reps, results=res), indent=1) + "\n")
    print(json.dumps(res, indent=1))
