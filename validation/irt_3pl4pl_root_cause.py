"""Pre-registered root-cause study of 3PL/4PL parameter recovery with Python mirt 1.2.0.

DESIGN (fixed before any result was seen; written into the output JSON):
  generator : independent NumPy logistic IRF (not mirt.simdata); truth as in
              validation/known_truth_irt_recovery.py extended to k = 30:
              a = linspace(.8, 1.6), b = linspace(-1.25, 1.25), c = linspace(.10, .18),
              d = linspace(.90, .97) (4PL only).
  theta     : N(0, 1); one extra condition with standardized chi-square(4) skewed theta.
  factors   : model {3PL, 4PL} x N {1000, 3000, 10000} x k {12, 30}; R = 20 seeds.
  estimators: ML21  = mirt default (EM, 21 GH nodes, tol 1e-4)          [PR #62 baseline]
              ML61  = EM, 61 GH nodes, tol 1e-6, max_iter 3000          [optimization adequacy]
              MAP   = mirt.estimation.priors.default_priors(model)      [library defaults,
                      a~LogN(0,.5), b~N(0,2), c~Beta(2,8), d~Beta(8,2); not tuned]
              TRUE0 = ML61 started at the true parameters (N=1000 only) [local-maximum check]
  per fit   : convergence, iterations, parameters at mirt box bounds, bias/RMSE per
              parameter, IRF RMSE (integrated over N(0,1)), IRF mapping check, and the
              marginal log-likelihood of estimate and truth on a common 61-node grid.
  pre-registered usability criteria (per condition, mean over converged fits):
              RMSE(a) <= .30, RMSE(b) <= .30, RMSE(c) <= .07, RMSE(d) <= .07,
              IRF RMSE <= .02 and <= 5 % of parameters at a box bound.
Failures and non-convergence are kept as records and never counted as measured.
Run: PYTHONPATH=. python validation/irt_3pl4pl_root_cause.py OUTDIR [--quick]
"""
import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from scipy.special import expit, logsumexp
from scipy.stats import norm

BOUNDS = {"a": (0.1, 5.0), "b": (-6.0, 6.0), "c": (0.0, 0.5), "d": (0.5, 1.0)}
CRITERIA = {"rmse_a": .30, "rmse_b": .30, "rmse_c": .07, "rmse_d": .07, "irf_rmse": .02, "at_bound_rate": .05}
NAMES = {"a": "discrimination", "b": "difficulty", "c": "guessing", "d": "upper"}


def truth(model, k):
    return {"a": np.linspace(.8, 1.6, k), "b": np.linspace(-1.25, 1.25, k),
            "c": np.linspace(.10, .18, k), "d": np.linspace(.90, .97, k) if model == "4PL" else np.ones(k)}


def irf(theta, p):
    return p["c"] + (p["d"] - p["c"]) * expit(p["a"] * (np.asarray(theta)[:, None] - p["b"]))


def simulate(model, n, k, seed, theta_dist="normal"):
    rng = np.random.default_rng(seed)
    if theta_dist == "normal":
        th = rng.standard_normal(n)
    else:  # standardized chi-square(4): skewness ~1.41
        th = (rng.chisquare(4, n) - 4) / np.sqrt(8)
    p = truth(model, k)
    return (rng.random((n, k)) < irf(th, p)).astype(int), p


def common_loglik(x, p, points=61):
    t = np.linspace(-6, 6, points); lw = norm.logpdf(t); lw -= logsumexp(lw)
    P = np.clip(irf(t, p), 1e-12, 1 - 1e-12)
    return float(logsumexp(x @ np.log(P).T + (1 - x) @ np.log1p(-P).T + lw, axis=1).sum())


def irf_rmse(est, tru):
    t = np.linspace(-4, 4, 161); w = norm.pdf(t); w /= w.sum()
    return float(np.sqrt(np.mean((w[:, None] * (irf(t, est) - irf(t, tru)) ** 2).sum(0))))


def fit_one(job):
    import mirt
    from mirt.estimation.priors import default_priors
    model, n, k, seed, estimator, theta_dist = job
    x, tru = simulate(model, n, k, seed, theta_dist)
    rec = {"model": model, "N": n, "k": k, "seed": seed, "estimator": estimator, "theta": theta_dist,
           "status": "FAILED"}
    kw = {"model": model}
    if estimator in ("ML61", "TRUE0"):
        kw.update(n_quadpts=61, tol=1e-6, max_iter=3000)
    if estimator == "MAP":
        kw["priors"] = default_priors(model)
    if estimator == "TRUE0":
        kw["start_values"] = {NAMES[q]: tru[q] for q in "abc" + ("d" if model == "4PL" else "")}
    t0 = time.perf_counter()
    try:
        r = mirt.fit_mirt(x, **kw)
        rec["seconds"] = time.perf_counter() - t0
        co = r.coef()
        est = {"a": co["discrimination"].to_numpy(float), "b": co["difficulty"].to_numpy(float),
               "c": co["guessing"].to_numpy(float),
               "d": co["upper"].to_numpy(float) if model == "4PL" else np.ones(k)}
        grid = np.linspace(-4, 4, 33)
        rec["irf_mapping_maxdiff"] = float(np.abs(np.asarray(r.model.probability(grid[:, None]), float)
                                                  .reshape(len(grid), -1) - irf(grid, est)).max())
        rec["converged"] = bool(r.converged); rec["iterations"] = int(r.n_iterations)
        free = "abc" + ("d" if model == "4PL" else "")
        at = {q: int(((est[q] - BOUNDS[q][0] <= 1e-3) | (BOUNDS[q][1] - est[q] <= 1e-3)).sum()) for q in free}
        rec["at_bound"] = at; rec["at_bound_rate"] = sum(at.values()) / (k * len(free))
        rec["metrics"] = {q: {"bias": float(np.mean(est[q] - tru[q])),
                              "rmse": float(np.sqrt(np.mean((est[q] - tru[q]) ** 2))),
                              "max_abs": float(np.max(np.abs(est[q] - tru[q])))} for q in free}
        rec["irf_rmse"] = irf_rmse(est, tru)
        rec["ll_est"] = common_loglik(x, est); rec["ll_true"] = common_loglik(x, tru)
        rec["ll_est_minus_true"] = rec["ll_est"] - rec["ll_true"]
        rec["corr_est_err_a_c"] = float(np.corrcoef(est["a"] - tru["a"], est["c"] - tru["c"])[0, 1])
        rec["status"] = "MEASURED" if rec["converged"] else "NONCONVERGED"
    except Exception as exc:  # recorded, never silently dropped
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return rec


def jobs(quick=False):
    seeds = range(1, 4) if quick else range(1, 21)
    out = []
    for model in ("3PL", "4PL"):
        for n in (1000, 3000, 10000):
            for k in (12, 30):
                for s in seeds:
                    base = 10_000 * (1 if model == "3PL" else 2) + 100 * k + s
                    for est in ("ML21", "ML61", "MAP") + (("TRUE0",) if n == 1000 else ()):
                        out.append((model, n, k, base + n, est, "normal"))
        for s in seeds:
            for est in ("ML21", "MAP"):
                out.append((model, 3000, 30, 900_000 + s, est, "skewed"))
    return out


def summarize(records):
    groups = {}
    for r in records:
        key = f"{r['model']}|N{r['N']}|k{r['k']}|{r['estimator']}|{r['theta']}"
        groups.setdefault(key, []).append(r)
    out = {}
    for key, rs in sorted(groups.items()):
        m = [r for r in rs if r["status"] == "MEASURED"]
        s = {"attempted": len(rs), "measured": len(m), "nonconverged": sum(r["status"] == "NONCONVERGED" for r in rs),
             "failed": sum(r["status"] == "FAILED" for r in rs)}
        if m:
            for q in m[0]["metrics"]:
                s[f"rmse_{q}"] = float(np.mean([r["metrics"][q]["rmse"] for r in m]))
                s[f"bias_{q}"] = float(np.mean([r["metrics"][q]["bias"] for r in m]))
                s[f"median_rmse_{q}"] = float(np.median([r["metrics"][q]["rmse"] for r in m]))
            s["irf_rmse"] = float(np.mean([r["irf_rmse"] for r in m]))
            s["at_bound_rate"] = float(np.mean([r["at_bound_rate"] for r in m]))
            s["fits_with_any_bound"] = int(sum(r["at_bound_rate"] > 0 for r in m))
            s["ll_est_minus_true_min"] = float(min(r["ll_est_minus_true"] for r in m))
            s["ll_est_minus_true_mean"] = float(np.mean([r["ll_est_minus_true"] for r in m]))
            s["seconds_mean"] = float(np.mean([r["seconds"] for r in m]))
            s["irf_mapping_maxdiff"] = float(max(r["irf_mapping_maxdiff"] for r in m))
            ok = all(s.get(f"rmse_{q}", 0) <= CRITERIA[f"rmse_{q}"] for q in "abcd" if f"rmse_{q}" in s)
            s["meets_preregistered_criteria"] = bool(ok and s["irf_rmse"] <= CRITERIA["irf_rmse"]
                                                     and s["at_bound_rate"] <= CRITERIA["at_bound_rate"]
                                                     and s["measured"] == s["attempted"])
        out[key] = s
    return out


def main(outdir, quick=False):
    os.makedirs(outdir, exist_ok=True)
    js = jobs(quick)
    t0 = time.time()
    with Pool(int(os.environ.get("P1_PROCS", "2"))) as pool:
        records = pool.map(fit_one, js, chunksize=1)
    payload = {"design": __doc__, "criteria": CRITERIA, "n_jobs": len(js), "wall_seconds": time.time() - t0,
               "versions": {"mirt": __import__("importlib.metadata").metadata.version("mirt"),
                            "numpy": np.__version__},
               "summary": summarize(records), "runs": records}
    with open(os.path.join(outdir, "irt_3pl4pl_root_cause.json"), "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, allow_nan=False)
    return payload


if __name__ == "__main__":
    main(sys.argv[1], quick="--quick" in sys.argv)
