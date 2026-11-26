"""Frozen vs optimised closure-K statistic and set equivalence (EQUIVALENCE_PROTOCOL.json)."""
import json, sys, importlib.util
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from validation.binary_certification.common import simulate, truth
from core.irt_python_mirt_adapter import fit_python_mirt
from core import irt_4pl_score_sets as NEW
spec = importlib.util.spec_from_file_location("frozen_k", "validation/p1_4pl_score_perf/frozen_k.py"); OLD = importlib.util.module_from_spec(spec); spec.loader.exec_module(OLD)


def grid_T(M, eta, x, t0):
    xx = np.asarray(x, int); N = len(xx); t, w = M._quad(); phat = np.bincount(xx @ M._BITS, minlength=4096) / N
    px, sc = (M.terms_fast if hasattr(M, "terms_fast") else M.terms)(eta, t, w); I = (px[:, None] * sc).T @ sc
    Hi = np.linalg.inv(I); anchor = np.clip(eta, M.LO, M.HI); out = []
    for j in range(12):
        for q in (2, 3):
            i = 4 * j + q; lo, hi = M.BOXQ[q]
            def T(psi):
                e = np.clip(anchor + Hi[:, i] * (psi - anchor[i]) / Hi[i, i], M.LO, M.HI); e[i] = np.clip(psi, M.LO[i], M.HI[i])
                return M.statistic(e, phat, i, N, t, w)
            out.append([T(p) for p in np.linspace(lo, hi, M.G)] + [T(t0[i])])
    return np.array(out)


def case(job):
    r, mode = job; x = simulate("4PL", 3000, 12, [202611262101, 4, r, 0])
    eta = fit_python_mirt(x, model="4PL", estimation="MAP", n_quadpts=61, max_iter=500, tol=1e-4).fit.coef()[["discrimination", "difficulty", "guessing", "upper"]].to_numpy(float).ravel()
    if mode == "c_low": eta[2::4] = 1e-6
    if mode == "c_high": eta[2::4] = .4999
    if mode == "d_low": eta[3::4] = .5 + 1e-6
    if mode == "d_high": eta[3::4] = 1 - 1e-6
    p = truth("4PL", 12); t0 = np.stack([p[s] for s in "abcd"], 1).ravel(); rec = {"r": r, "mode": mode}
    try:
        a = OLD.confidence_sets(eta, x, truth=t0); rec["frozen_refused"] = False
    except np.linalg.LinAlgError: a = None; rec["frozen_refused"] = True
    try:
        b = NEW.confidence_sets(eta, x, truth=t0); rec["new_refused"] = False
    except np.linalg.LinAlgError: b = None; rec["new_refused"] = True
    if a is None or b is None: rec["equal"] = a is None and b is None; return rec
    To, Tn = grid_T(OLD, eta, x, t0), grid_T(NEW, eta, x, t0)
    rec.update(max_rel_T=float(np.max(np.abs(Tn - To) / np.maximum(1, To))), near_ties=int(np.sum(np.abs(To - OLD.CRIT) <= 1e-6)),
               decisions_equal=bool(np.array_equal(To <= OLD.CRIT, Tn <= NEW.CRIT)), sets_equal=a == b,
               topology={k: sum(s[k] for s in a) for k in ("whole", "empty")} | {"split": sum(len(s["intervals"]) > 1 for s in a)})
    rec["equal"] = rec["decisions_equal"] and rec["sets_equal"] and rec["max_rel_T"] <= 1e-8
    return rec


if __name__ == "__main__":
    jobs = [(r, "fit") for r in range(8)] + [(r, m) for r in range(4) for m in ("c_low", "c_high", "d_low", "d_high")]
    with ProcessPoolExecutor(2) as ex: res = list(ex.map(case, jobs))
    rep = {"cases": res, "all_equal": all(r["equal"] for r in res)}
    json.dump(rep, open(sys.argv[1], "w"), indent=1); print(rep["all_equal"], max(r.get("max_rel_T", 0) for r in res), [r.get("topology") for r in res[:8]], sum(r.get("near_ties", 0) for r in res), [(r["mode"], r["frozen_refused"]) for r in res if r["frozen_refused"]])
