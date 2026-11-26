"""Finite-sample lower bound on the expected length of ANY confidence interval for one item parameter.

For C with coverage >= 1-alpha at every parameter value (nuisance free): E_0 len(C) = int P_0(t in C) dt and
P_0(t in C) >= 1-alpha - TV(P_0^N, P_t^N) for any eta_t with that coordinate = t (Pratt-type argument).
TV bound: min(Pinsker sqrt(N KL/2), sqrt(1 - BC^(2N))) with exact 4096-pattern marginals; eta_t = argmin KL with the
coordinate fixed (any choice is valid; KL minimisation only tightens). Exact, design-specific, no simulation.
"""
import json, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import minimize
from validation.binary_certification.common import truth
from validation.p3_sx2_analytic import kernel as K

BOX = {2: (1e-6, .5), 3: (.5, 1 - 1e-6)}


def task(args):
    model, j, q, grid = args
    k = 4 if model == "4PL" else 3; p = truth(model, 12); e0 = np.stack([p[s] for s in "abcd"[:k]], 1).ravel()
    p0, _ = K.pattern_terms(e0, model); idx = j * k + q
    free = np.array([i for i in range(len(e0)) if i != idx])
    bnds = [((.05, 6) if i % k == 0 else (-6, 6) if i % k == 1 else BOX[i % k]) for i in free]
    out = []; start = e0[free].copy()
    for t in sorted(grid, key=lambda v: abs(v - e0[idx])):
        def f(z):
            e = e0.copy(); e[free] = z; e[idx] = t; px, sc = K.pattern_terms(e, model)
            return float(p0 @ (np.log(p0) - np.log(px))), -(p0 @ sc)[free]
        r = minimize(f, start, jac=True, method="L-BFGS-B", bounds=bnds, options={"maxiter": 150, "ftol": 1e-15, "gtol": 1e-10})
        e = e0.copy(); e[free] = r.x; e[idx] = t; px, _ = K.pattern_terms(e, model)
        out.append(dict(t=float(t), kl=max(float(r.fun), 0.0), bc=float(np.sum(np.sqrt(p0 * px))), converged=bool(r.success), nit=int(r.nit)))
    return dict(model=model, item=j, param="abcd"[q], truth=float(e0[idx]), points=sorted(out, key=lambda d: d["t"]))


def length_bound(res, N, alpha=.05):
    t = np.array([d["t"] for d in res["points"]]); kl = np.array([d["kl"] for d in res["points"]]); bc = np.array([d["bc"] for d in res["points"]])
    tv = np.minimum(np.minimum(np.sqrt(N * kl / 2), np.sqrt(np.clip(1 - bc**(2 * N), 0, 1))), 1)
    return float(np.trapezoid(np.maximum(0, 1 - alpha - tv), t))   # trapezoid between grid points (KL smooth); exact-grid integral


if __name__ == "__main__":
    jobs = []
    for model, q in (("4PL", 2), ("4PL", 3), ("3PL", 2)):
        lo, hi = BOX[q]
        for j in (0, 3, 6, 9, 11): jobs.append((model, j, q, np.r_[np.linspace(lo, hi, 11), truth(model, 12)["abcd"[q]][j]]))
    with ProcessPoolExecutor(int(sys.argv[2]) if len(sys.argv) > 2 else 2) as ex:
        res = list(ex.map(task, jobs))
    for r in res:
        r["bound_N3000"] = length_bound(r, 3000); r["box"] = BOX["abcd".index(r["param"])][1] - BOX["abcd".index(r["param"])][0]
        r["N_for_bound_le_half_box"] = next((int(n) for n in np.geomspace(1000, 1e7, 200) if length_bound(r, n) <= r["box"] / 2), None)
    json.dump(res, open(sys.argv[1], "w"), indent=1)
    for r in res: print(r["model"], r["param"], r["item"], round(r["bound_N3000"], 3), r["N_for_bound_le_half_box"], all(p["converged"] for p in r["points"]))
