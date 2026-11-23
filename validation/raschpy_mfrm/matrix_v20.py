"""V20: matrix-only confirmatory recovery with Monte Carlo precision for the 0.05 bias gate (PROTOCOL V20).
PYTHONPATH=. python -m validation.raschpy_mfrm.matrix_v20 [--reps 5000] [--workers 4] [--design D --scenario S --out F]
"""
import argparse, json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import validation.raschpy_mfrm.familywise_v19 as V
from validation.raschpy_mfrm.familywise_v18 import evaluate

H = Path(__file__).resolve().parent; SEED = 20261127; CAL = 1000; FOCUS = "cell:('P5', 'K3'):3"


def job(args):
    design, rep, scenario = args; rows = []
    lam, c, om = V.truth("matrix", design, scenario)
    d = V.simulate(design, c, np.random.default_rng([SEED, scenario == "nonzero", list(V.DESIGNS).index(design), rep]))
    try:
        for k, (e, t, s) in V.fit_params(d, "matrix", design, lam, c, om).items():
            rows.append(dict(design=design, rep=rep, model="matrix", key=k, est=e, truth=t, se=s))
    except Exception as ex:
        rows.append(dict(design=design, rep=rep, model="matrix", key="FIT_ERROR", est=np.nan, truth=np.nan, se=np.nan, err=str(ex)[:80]))
    return rows


def bias_ci(df, key):
    g = df[(df.key == key) & (df.rep >= CAL)]; dev = g.est - g.truth
    m, se = float(dev.mean()), float(dev.std() / np.sqrt(len(dev)))
    return dict(key=key, bias=m, mcse=se, ci95=[m - 1.959964 * se, m + 1.959964 * se], n=int(len(dev)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=5000); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--design", choices=list(V.DESIGNS)); ap.add_argument("--scenario", choices=["null", "nonzero"])
    ap.add_argument("--out", default=str(H / "matrix_v20.json")); a = ap.parse_args()
    designs = [a.design] if a.design else list(V.DESIGNS); scens = [a.scenario] if a.scenario else ["null", "nonzero"]
    res = {}
    with Pool(a.workers) as p:
        for scen in scens:
            df = pd.DataFrame([r for o in p.imap_unordered(job, [(d, r, scen) for d in designs for r in range(a.reps)], chunksize=25) for r in o])
            ev = evaluate(df, 2 * CAL)["matrix"]                            # calibration = reps < CAL, confirmatory = rest
            for d in designs:
                sub = df[(df.design == d) & (df.key != "FIT_ERROR")]
                per = sub[sub.rep >= CAL].assign(dev=lambda x: x.est - x.truth).groupby("key").dev.mean()
                ev[d]["worst_cell"] = bias_ci(sub, per.abs().idxmax()); ev[d]["focus_cell"] = bias_ci(sub, FOCUS)
            res[scen] = {d: ev[d] for d in designs}
    if not a.design and not a.scenario:
        res["pass_all"] = all(res[s][d]["pass"] for s in scens for d in designs)
    Path(a.out).write_text(json.dumps(dict(protocol="PROTOCOL.json V20_matrix_confirmatory", seed=SEED, reps=a.reps, calibration=CAL,
                                           results=res), indent=1) + "\n")
    print(json.dumps({s: {d: {k: res[s][d][k] for k in ("max_abs_bias", "pooled_coverage", "familywise_recovery", "pass", "worst_cell", "focus_cell")}
                          for d in designs} for s in scens}, indent=1))
