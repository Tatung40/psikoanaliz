"""Deterministic MFRM fixtures with known generating parameters (no personal data; synthetic IDs).
Rating-scale many-facet Rasch model: log(P_k / P_{k-1}) = theta_n - delta_i - lambda_j (- gamma_t) - tau_k.
Run: PYTHONPATH=. python examples/mfrm/build.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEED = 20261107
TAU = np.array([-1.2, 0.0, 1.2])                       # 4 ordered categories 0..3, thresholds sum to zero
ITEMS = {"K1": -0.5, "K2": -0.1, "K3": 0.2, "K4": 0.4}
RATERS = {"P1": -0.8, "P2": 0.0, "P3": 0.8}            # lenient, average, severe (sum to zero)


def _score(rng, eta, tau=TAU):
    num = np.concatenate([[0.], np.cumsum(eta - tau)])
    p = np.exp(num - num.max()); p /= p.sum()
    return int(rng.choice(len(p), p=p))


def _rows(rng, persons, pairs, theta, items=ITEMS, raters=RATERS, tasks=None, bias=None, tau=TAU):
    out = []
    for n, pid in enumerate(persons):
        for i, d in items.items():
            for j in pairs(n):
                for t, g in (tasks or {None: 0.}).items():
                    eta = theta[n] - d - raters[j] - g - (bias or {}).get((j, i), 0.)
                    row = {"person_id": pid, "item_id": i, "rater_id": j, "score": _score(rng, eta, tau)}
                    if t is not None:
                        row["task_id"] = t
                    out.append(row)
    return pd.DataFrame(out)


def fixtures():
    f = {}; truth = {}
    r = np.random.default_rng(SEED); n = 300; ids = [f"S{k:04d}" for k in range(n)]; th = r.normal(0, 1, n)
    f["balanced_3facet"] = _rows(r, ids, lambda k: list(RATERS), th)
    truth["balanced_3facet"] = dict(items=ITEMS, raters=RATERS, thresholds=TAU.tolist(), sigma=1.0)
    raters5 = {"P1": -0.8, "P2": -0.3, "P3": 0.0, "P4": 0.3, "P5": 0.8}
    r = np.random.default_rng(SEED + 1); n = 400; ids = [f"U{k:04d}" for k in range(n)]; th = r.normal(0, 1, n)
    f["unbalanced_linked"] = _rows(r, ids, lambda k: [f"P{k % 5 + 1}", f"P{(k + 1) % 5 + 1}"], th, raters=raters5)
    truth["unbalanced_linked"] = dict(items=ITEMS, raters=raters5, thresholds=TAU.tolist(), sigma=1.0,
                                      design="each person rated by 2 of 5 raters in a cyclic (connected) rotation")
    r = np.random.default_rng(SEED + 2); n = 250; ids = [f"T{k:04d}" for k in range(n)]; th = r.normal(0, 1, n)
    tasks = {"G1": -0.3, "G2": 0.3}
    f["four_facet"] = _rows(r, ids, lambda k: list(RATERS), th, tasks=tasks)
    truth["four_facet"] = dict(items=ITEMS, raters=RATERS, tasks=tasks, thresholds=TAU.tolist(), sigma=1.0)
    r = np.random.default_rng(SEED + 3); n = 300; ids = [f"B{k:04d}" for k in range(n)]; th = r.normal(0, 1, n)
    f["interaction_bias"] = _rows(r, ids, lambda k: list(RATERS), th, bias={("P2", "K3"): 1.0})
    truth["interaction_bias"] = dict(items=ITEMS, raters=RATERS, thresholds=TAU.tolist(), sigma=1.0,
                                     interaction={"P2:K3": 1.0})
    r = np.random.default_rng(SEED + 4); n = 200; ids = [f"Y{k:04d}" for k in range(n)]; th = r.normal(0, 1, n)
    f["sparse_category"] = _rows(r, ids, lambda k: list(RATERS), th, tau=np.array([-1.0, 0.2, 3.5]))
    truth["sparse_category"] = dict(note="top category rarely used (tau3 = 3.5); category diagnostics must warn")
    r = np.random.default_rng(SEED + 5); n = 120; ids = [f"D{k:04d}" for k in range(n)]; th = r.normal(0, 1, n)
    f["disconnected"] = _rows(r, ids, lambda k: ["P1", "P2"] if k < 60 else ["P3", "P4"], th,
                              raters={"P1": -.5, "P2": 0., "P3": .2, "P4": .3})
    truth["disconnected"] = dict(note="persons 0-59 rated only by P1/P2, persons 60-119 only by P3/P4: not connected")
    bad = f["balanced_3facet"].head(40).copy()
    bad = pd.concat([bad, bad.iloc[[0]]]); bad.iloc[5, bad.columns.get_loc("score")] = np.nan
    f["invalid_duplicate_missing"] = bad.reset_index(drop=True)
    truth["invalid_duplicate_missing"] = dict(note="one duplicated person-item-rater row and one missing score")
    return f, truth


if __name__ == "__main__":
    f, truth = fixtures(); meta = {}
    for name, d in f.items():
        p = HERE / f"{name}.csv"; d.to_csv(p, index=False)
        meta[name] = dict(file=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest(), rows=len(d), truth=truth[name])
    with pd.ExcelWriter(HERE / "mfrm_examples.xlsx") as w:
        for name, d in f.items():
            d.to_excel(w, sheet_name=name[:31], index=False)
    (HERE / "fixtures.json").write_text(json.dumps(dict(seed=SEED, model="RSM-MFRM", fixtures=meta), indent=1) + "\n")
    print({k: v["rows"] for k, v in meta.items()})
