"""Pre-registered RSM recovery, SE calibration, CI coverage and RSM-vs-PCM LR study.

DESIGN (fixed before results):
  generator : independent NumPy RSM sampler (inverse-CDF on category probabilities), C = 5,
              delta = linspace(-1.5, 1.5, k), tau = (-1.2, -0.3, 0.4, 1.1) (sum 0), sigma = 1.2.
  factors   : N {300, 1000, 3000} x k {10, 20}; R = 100 seeds; identification "rasch" (sigma free).
  per fit   : convergence, bias/RMSE of delta, tau, sigma; Wald 95% CI coverage; mean SE vs empirical SD.
  LR study  : RSM vs PCM nested LR under (i) RSM truth (Type I) and (ii) PCM truth where items 1..k/5
              have tau + (0.6, -0.6, -0.6, 0.6) (power); N {300, 1000}, k = 10, R = 100.
  library   : 20 replications N = 1000, k = 10, sigma = 1 cross-checked against Python mirt
              EMEstimator(RatingScaleModel) under the unit-variance identification (61 nodes):
              |logLik difference| and max |parameter difference| after the documented mapping.
  criteria  : at N >= 1000 mean CI coverage per parameter block in [.925, .975]; |mean bias| <= .05;
              RMSE non-increasing in N; LR Type I in [.025, .075]; library |dlogLik| < 1e-4 and
              max parameter difference < 1e-3 (pre-set numerical tolerances, not tuned).
Run: PYTHONPATH=. python validation/rsm_recovery_study.py OUTDIR [--quick]
"""
import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd

TAU = np.array([-1.2, -.3, .4, 1.1]); SD = 1.2
Z = 1.959963984540054


def recovery_job(job):
    from core import irt_rsm as R
    n, k, seed = job
    delta = np.linspace(-1.5, 1.5, k)
    rec = {"type": "recovery", "N": n, "k": k, "seed": seed, "status": "FAILED"}
    try:
        x = R.simulate(delta, TAU, n, SD, seed)
        f = R.fit(pd.DataFrame(x))
        rec["converged"] = f["converged"]
        est = {"delta": f["delta"], "tau": f["tau"], "sigma": np.array([f["sd"]])}
        tru = {"delta": delta, "tau": TAU, "sigma": np.array([SD])}
        t = f["tables"]
        se = {"delta": t["items"]["SH(δ)"].to_numpy(float), "tau": t["steps"]["SH(τ)"].to_numpy(float),
              "sigma": t["latent"]["SH(σ)"].to_numpy(float)}
        rec["est"] = {p: v.tolist() for p, v in est.items()}
        rec["se"] = {p: v.tolist() for p, v in se.items()}
        rec["covered"] = {p: (np.abs(est[p] - tru[p]) <= Z * se[p]).astype(int).tolist() for p in est}
        rec["status"] = "MEASURED" if f["converged"] and np.isfinite(np.r_[se["delta"], se["tau"], se["sigma"]]).all() else "NONCONVERGED"
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return rec


def lr_job(job):
    from core import irt_rsm as R
    truth, n, k, seed = job
    rec = {"type": "lr", "truth": truth, "N": n, "k": k, "seed": seed, "status": "FAILED"}
    try:
        delta = np.linspace(-1.5, 1.5, k)
        if truth == "RSM":
            x = R.simulate(delta, TAU, n, SD, seed)
        else:
            rng = np.random.default_rng(seed); th = SD * rng.standard_normal(n)
            steps = R.rsm_steps(delta, TAU)
            steps[: k // 5] += np.array([.6, -.6, -.6, .6])
            P = np.exp(R.category_log_probs(th, steps))
            x = (rng.random((n, k))[..., None] > np.cumsum(P, 2)).sum(2)
        out = R.rsm_vs_pcm(pd.DataFrame(x))
        rec.update(LR=out["LR"], df=out["df"], p=out["p"], valid=out["valid"])
        rec["status"] = "MEASURED" if out["valid"] else "NONCONVERGED"
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return rec


def library_job(seed):
    from core import irt_rsm as R
    from mirt import EMEstimator, RatingScaleModel
    rec = {"type": "library", "seed": seed, "status": "FAILED"}
    try:
        delta = np.linspace(-1.5, 1.5, 10)
        x = R.simulate(delta, TAU, 1000, 1.0, seed)
        f = R.fit(pd.DataFrame(x), identification="unit_variance", points=61)
        m = RatingScaleModel(n_items=10, n_categories=5)
        res = EMEstimator(n_quadpts=61, tol=1e-8, max_iter=5000).fit(m, x)
        mp = R.to_mirt_parameterization(f["delta"], f["tau"])
        rec.update(dll=float(f["loglik"] - res.log_likelihood),
                   dpar=float(max(np.abs(mp["difficulty"] - m._parameters["difficulty"]).max(),
                                  np.abs(mp["thresholds"] - m._parameters["thresholds"]).max())),
                   native_converged=f["converged"], mirt_converged=bool(res.converged))
        rec["status"] = "MEASURED"
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return rec


def dispatch(job):
    kind, payload = job
    return {"recovery": recovery_job, "lr": lr_job, "library": library_job}[kind](payload)


def jobs(quick=False):
    R_ = 4 if quick else 100
    out = [("recovery", (n, k, 50_000 + 1000 * k + n + r)) for n in (300, 1000, 3000) for k in (10, 20) for r in range(R_)]
    out += [("lr", (t, n, 10, 70_000 + 10_000 * (t == "PCM") + n + r)) for t in ("RSM", "PCM") for n in (300, 1000)
            for r in range(R_)]
    out += [("library", 90_000 + r) for r in range(3 if quick else 20)]
    return out


def summarize(recs):
    from scipy.stats import beta as B
    def cp(x, n):
        return [float(B.ppf(.025, x, n - x + 1)) if x else 0., float(B.ppf(.975, x + 1, n - x)) if x < n else 1.]
    out = {"recovery": [], "lr": [], "library": {}}
    for n in (300, 1000, 3000):
        for k in (10, 20):
            rs = [r for r in recs if r["type"] == "recovery" and r["N"] == n and r["k"] == k]
            m = [r for r in rs if r["status"] == "MEASURED"]
            row = {"N": n, "k": k, "attempted": len(rs), "measured": len(m), "failed": sum(r["status"] == "FAILED" for r in rs)}
            if m:
                delta = np.linspace(-1.5, 1.5, k); tru = {"delta": delta, "tau": TAU, "sigma": np.array([SD])}
                for p in ("delta", "tau", "sigma"):
                    E = np.array([r["est"][p] for r in m]); S = np.array([r["se"][p] for r in m])
                    cov = np.array([r["covered"][p] for r in m])
                    row[f"bias_{p}"] = float((E - tru[p]).mean()); row[f"max_abs_bias_{p}"] = float(np.abs((E - tru[p]).mean(0)).max())
                    row[f"rmse_{p}"] = float(np.sqrt(((E - tru[p]) ** 2).mean()))
                    row[f"coverage_{p}"] = float(cov.mean()); row[f"coverage_{p}_ci"] = cp(int(cov.sum()), cov.size)
                    row[f"se_ratio_{p}"] = float((S.mean(0) / E.std(0, ddof=1)).mean())
            out["recovery"].append(row)
    for t in ("RSM", "PCM"):
        for n in (300, 1000):
            rs = [r for r in recs if r["type"] == "lr" and r["truth"] == t and r["N"] == n]
            m = [r for r in rs if r["status"] == "MEASURED"]
            rej = int(sum(r["p"] < .05 for r in m))
            out["lr"].append({"truth": t, "N": n, "attempted": len(rs), "measured": len(m),
                              "rejection_rate": rej / len(m) if m else np.nan, "ci": cp(rej, len(m)) if m else None,
                              "df": m[0]["df"] if m else None})
    lib = [r for r in recs if r["type"] == "library" and r["status"] == "MEASURED"]
    out["library"] = {"attempted": sum(r["type"] == "library" for r in recs), "measured": len(lib),
                      "max_abs_dloglik": float(max(abs(r["dll"]) for r in lib)) if lib else np.nan,
                      "max_param_diff": float(max(r["dpar"] for r in lib)) if lib else np.nan,
                      "all_converged": bool(all(r["native_converged"] and r["mirt_converged"] for r in lib))}
    return out


def main(outdir, quick=False):
    os.makedirs(outdir, exist_ok=True)
    js = jobs(quick); t0 = time.time()
    with Pool(int(os.environ.get("P3_PROCS", "2"))) as pool:
        recs = pool.map(dispatch, js, chunksize=1)
    payload = {"design": __doc__, "n_jobs": len(js), "wall_seconds": time.time() - t0,
               "summary": summarize(recs), "runs": recs}
    with open(os.path.join(outdir, "rsm_recovery_study.json"), "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, default=float)
    return payload


if __name__ == "__main__":
    main(sys.argv[1], quick="--quick" in sys.argv)
