"""Amendment D1: 4PL independent validation pathway on 8 fresh holdout fixtures (no R, no Python mirt in the reference).

    PYTHONPATH=. python validation/binary_certification/d1_4pl.py --workers 2 --output validation/evidence/binary_certification/d1_4pl.json
"""
import argparse
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

D1 = json.loads((Path(__file__).with_name("PROTOCOL.json")).read_text())["amendment_D"]["D1_4PL_independent_pathway"]
SIM = re.compile(r"sim\(N=(\d+), k=(\d+), seed=(\d+)\)")
PERTURB = [(.6, .05, .85, -.5), (1.8, .3, .99, .5), (.6, .3, .99, .5), (1.8, .05, .85, -.5)]


def one(fixture):
    from core.irt_python_mirt_adapter import fit_python_mirt
    from validation.binary_certification.b2_4pl import _agree
    from validation.binary_certification.common import Objective, simulate
    from validation.binary_certification.reference_numpy import BOX, data_start, fit, objective
    n, k, seed = map(int, SIM.fullmatch(fixture).groups())
    x = simulate("4PL", n, k, seed); obj = Objective(x, "4PL", map_priors=True)
    py = fit_python_mirt(x, model="4PL", estimation="MAP", n_quadpts=61, tol=1e-8, max_iter=5000).fit; co = py.coef()
    P = {"a": co.discrimination.to_numpy(float), "b": co.difficulty.to_numpy(float), "c": co.guessing.to_numpy(float), "d": co.upper.to_numpy(float)}
    base = data_start(x); starts = [base] + [{"a": np.full(k, a), "b": base["b"] + db, "c": np.full(k, c), "d": np.full(k, d)}
                                             for a, c, d, db in PERTURB]
    runs = []
    for s in starts:
        Q, info = fit(x, "4PL", start=s); runs.append((obj.value(Q), Q, info))
    best = max(runs, key=lambda r: r[0])
    agree, diff, gap = _agree(obj, P, best[1])
    f_g, qs, _ = objective(x, "4PL")
    v = np.ravel(np.column_stack([P[q] for q in qs])); _, g = f_g(v)
    lo = np.array([BOX[q][0] for _ in range(k) for q in qs]); hi = np.array([BOX[q][1] for _ in range(k) for q in qs])
    at_lo, at_hi = v - lo <= 1e-6, hi - v <= 1e-6; interior = ~(at_lo | at_hi)
    kkt = bool(np.abs(g[interior]).max() <= .05 and np.all(g[at_lo] >= 0) and np.all(g[at_hi] <= 0))   # g = gradient of -logpost
    idx = np.flatnonzero(interior); h = 1e-4; H = np.zeros((len(idx), len(idx)))
    for j, i in enumerate(idx):
        e = np.zeros_like(v); e[i] = h
        H[:, j] = (f_g(v + e)[1][idx] - f_g(v - e)[1][idx]) / (2 * h)
    min_eig = float(np.linalg.eigvalsh((H + H.T) / 2)[0])
    return {"fixture": fixture, "python_converged": bool(py.converged), "starts": [{"objective": r[0], **r[2]} for r in runs],
            "starts_at_best": int(sum(abs(r[0] - best[0]) <= 1e-3 for r in runs)), "agreement": agree, "max_abs_diff": diff,
            "objective_gap_python_minus_best": gap, "max_abs_gradient_interior": float(np.abs(g[interior]).max()),
            "bound_active": int((~interior).sum()), "kkt": kkt, "hessian_min_eigenvalue": min_eig,
            "pass": bool(py.converged and agree and kkt and min_eig > 0)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=2); ap.add_argument("--output", required=True)
    a = ap.parse_args(); t = time.time()
    order = sorted(D1["fresh_holdout"], key=lambda f: -int(SIM.fullmatch(f).group(1)) * int(SIM.fullmatch(f).group(2)))
    with ProcessPoolExecutor(a.workers) as ex:
        done = dict(zip(order, ex.map(one, order)))
    rows = [done[f] for f in D1["fresh_holdout"]]
    res = {"protocol": "PROTOCOL.json amendment_D D1", "fixtures": rows, "all_pass": all(r["pass"] for r in rows), "wall_seconds": round(time.time() - t)}
    Path(a.output).write_text(json.dumps(res, indent=1) + "\n")
    for r in rows:
        print(r["fixture"], r["pass"], "starts@best", r["starts_at_best"], {q: f"{v:.1e}" for q, v in r["max_abs_diff"].items()},
              f"gap {r['objective_gap_python_minus_best']:.1e} |g| {r['max_abs_gradient_interior']:.1e} eig {r['hessian_min_eigenvalue']:.2f}")
