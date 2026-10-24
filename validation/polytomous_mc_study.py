"""Pre-registered Monte Carlo for Python mirt 1.2.0 GRM / PCM / GPCM / NRM (single group).

DESIGN (fixed before results):
  generator : independent NumPy samplers using the closed forms verified against mirt.probability:
              GRM  P(X>=c) = logistic(a(theta - b_c)), a = linspace(.8, 2.0, k), b_j = loc_j + (-1, 0, 1)
              PCM  P ~ exp(sum_{s<=c} (theta - b_s)), b_j = loc_j + (-.8, 0, .8)
              GPCM P ~ exp(sum_{s<=c} a(theta - b_s)), a = linspace(.7, 1.7, k), same steps as PCM
              NRM  P ~ exp(A_c theta + D_c), A_j = s_j * (0, 1, 2, 3), s_j = linspace(.6, 1.2, k),
                   D_j = (0, .5, .3, -.4) + loc_j * (0, -1, -2, -3)  (first category reference)
              loc = linspace(-1, 1, k); theta ~ N(0, 1); k = 10; C = 4.
  factors   : model x N {500, 2000}; R = 50 seeds.
  per fit   : convergence; parameter bias/RMSE per block (NRM after reference-category centring and
              sign orientation by item 1); Wald 95% CI coverage with mirt SEs; category-curve RMSE over
              N(0,1); EAP theta correlation/RMSE vs true theta; information check: mirt
              model.information vs independent sum_c (dP_c/dtheta)^2 / P_c (finite differences).
  model choice (N = 2000): GRM data also fit by GPCM and vice versa; AIC/BIC choice of the true model.
  criteria  : converged >= 95 %; curve RMSE <= .02 at N = 2000; mean CI coverage per block in
              [.925, .975] at N = 2000; information max relative difference <= 1e-4;
              EAP correlation >= .85. Failures are recorded and never counted as measured.
Run: PYTHONPATH=. python validation/polytomous_mc_study.py OUTDIR [--quick]
"""
import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.special import expit, softmax
from scipy.stats import norm

K, C = 10, 4
LOC = np.linspace(-1, 1, K)
ENGINE = os.environ.get("P4_ENGINE", "library_default")  # "adapter" = the application's fit path
CRITERIA = {"converged_rate": .95, "curve_rmse": .02, "coverage": (.925, .975), "info_rel": 1e-4, "eap_corr": .85}


def truth(model):
    if model == "GRM":
        return {"discrimination": np.linspace(.8, 2.0, K), "thresholds": LOC[:, None] + np.array([-1., 0., 1.])}
    if model == "PCM":
        return {"steps": LOC[:, None] + np.array([-.8, 0., .8])}
    if model == "GPCM":
        return {"discrimination": np.linspace(.7, 1.7, K), "steps": LOC[:, None] + np.array([-.8, 0., .8])}
    s = np.linspace(.6, 1.2, K)
    return {"slopes": s[:, None] * np.arange(C)[None, :],
            "intercepts": np.array([0., .5, .3, -.4])[None, :] + LOC[:, None] * -np.arange(C)[None, :]}


def probs(model, p, theta):
    th = np.asarray(theta, float)[:, None, None]
    if model == "GRM":
        a, b = p["discrimination"], p["thresholds"]
        cum = np.concatenate([np.ones((len(theta), K, 1)), expit(a[None, :, None] * (th - b[None])),
                              np.zeros((len(theta), K, 1))], 2)
        return cum[..., :-1] - cum[..., 1:]
    if model in ("PCM", "GPCM"):
        a = p.get("discrimination", np.ones(K)); b = p["steps"]
        eta = np.concatenate([np.zeros((len(theta), K, 1)), np.cumsum(a[None, :, None] * (th - b[None]), 2)], 2)
        return softmax(eta, 2)
    return softmax(p["slopes"][None] * th + p["intercepts"][None], 2)


def simulate(model, n, seed):
    rng = np.random.default_rng(seed); th = rng.standard_normal(n)
    P = probs(model, truth(model), th)
    return (rng.random((n, K))[..., None] > np.cumsum(P, 2)).sum(2), th


def _orient_nrm(p):
    A = np.asarray(p["slopes"], float); D = np.asarray(p["intercepts"], float)
    A = A - A[:, :1]; D = D - D[:, :1]
    if A[0, -1] < 0:
        A = -A
    return {"slopes": A, "intercepts": D}


def fit_job(job):
    import mirt
    model, n, seed = job
    rec = {"model": model, "N": n, "seed": seed, "status": "FAILED"}
    t0 = time.perf_counter()
    try:
        x, th = simulate(model, n, seed)
        if ENGINE == "adapter":
            from core.irt_python_mirt_adapter import fit_python_mirt
            r = fit_python_mirt(x, model=model, n_categories=C).fit
        else:
            r = mirt.fit_mirt(x, model=model, n_categories=C)
        rec["seconds"] = time.perf_counter() - t0
        rec["converged"] = bool(r.converged)
        est = {k: np.asarray(v, float) for k, v in r.model.parameters.items()}
        se = {k: np.asarray(v, float) for k, v in r.standard_errors.items()}
        tru = truth(model)
        if model == "NRM":
            flip = np.asarray(est["slopes"])[0, -1] - np.asarray(est["slopes"])[0, 0] < 0
            est = _orient_nrm(est)
            free = (slice(None), slice(1, None))  # reference category is fixed, not estimated
        else:
            free = (slice(None), slice(None))
        grid = np.linspace(-4, 4, 81)
        P_lib = np.asarray(r.model.probability(grid[:, None]), float)
        rec["mapping_maxdiff"] = float(np.abs(P_lib - probs(model, {k: np.asarray(v, float) for k, v in r.model.parameters.items()}, grid)).max())
        w = norm.pdf(grid); w /= w.sum()
        rec["curve_rmse"] = float(np.sqrt((w[:, None, None] * (probs(model, est, grid) - probs(model, tru, grid)) ** 2).sum(0).mean()))
        rec["blocks"] = {}
        for name, tv in tru.items():
            ev = est[name]
            if name == "discrimination" and model == "PCM":
                continue
            sl = free if model == "NRM" else (slice(None),) * ev.ndim
            e, t = ev[sl], tv[sl]
            s = np.asarray(se[name], float)[sl] if name in se else np.full(e.shape, np.nan)
            ok = np.isfinite(s) & (s > 0)
            rec["blocks"][name] = {"bias": float((e - t).mean()), "rmse": float(np.sqrt(((e - t) ** 2).mean())),
                                   "covered": int((np.abs(e - t) <= 1.96 * s)[ok].sum()), "n_ci": int(ok.sum()),
                                   "n_param": int(e.size)}
        h = 1e-5; Pp = probs(model, est, grid + h); Pm = probs(model, est, grid - h); P0 = probs(model, est, grid)
        info_ind = (((Pp - Pm) / (2 * h)) ** 2 / np.maximum(P0, 1e-300)).sum(2)
        info_lib = np.column_stack([np.asarray(r.model.information(grid[:, None], item_idx=j), float).reshape(-1)
                                    for j in range(K)])
        rec["test_info_rel_maxdiff"] = float((np.abs(np.asarray(r.model.information(grid[:, None]), float).reshape(-1)
                                                  - info_ind.sum(1)) / info_ind.sum(1)).max())
        rec["info_rel_maxdiff"] = float((np.abs(info_lib - info_ind) / np.maximum(info_ind, 1e-8)).max())
        sc = mirt.fscores(r, x, method="EAP")
        eap = np.asarray(sc.theta, float).reshape(-1)
        if model == "NRM" and flip:
            eap = -eap
        rec["eap_corr"] = float(np.corrcoef(eap, th)[0, 1]); rec["eap_rmse"] = float(np.sqrt(np.mean((eap - th) ** 2)))
        rec["mean_eap_se"] = float(np.nanmean(np.asarray(sc.standard_error, float)))
        if n == 2000 and model in ("GRM", "GPCM"):
            other = "GPCM" if model == "GRM" else "GRM"
            r2 = mirt.fit_mirt(x, model=other, n_categories=C)
            rec["choice"] = {"aic_true": bool(r.aic < r2.aic), "bic_true": bool(r.bic < r2.bic),
                             "other_converged": bool(r2.converged)}
        rec["status"] = "MEASURED" if rec["converged"] else "NONCONVERGED"
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return rec


def jobs(quick=False):
    R = 3 if quick else 50
    models = [m for m in os.environ.get("P4_MODELS", "GRM,PCM,GPCM,NRM").split(",")]
    order = ("GRM", "PCM", "GPCM", "NRM")
    return [(m, n, 300_000 + 10_000 * order.index(m) + n + r) for m in models for n in (500, 2000) for r in range(R)]


def summarize(recs):
    from scipy.stats import beta as B
    def cp(x, n):
        return [float(B.ppf(.025, x, n - x + 1)) if x else 0., float(B.ppf(.975, x + 1, n - x)) if x < n else 1.]
    out = []
    for m in ("GRM", "PCM", "GPCM", "NRM"):
        for n in (500, 2000):
            rs = [r for r in recs if r["model"] == m and r["N"] == n]
            if not rs:
                continue
            ms = [r for r in rs if r["status"] == "MEASURED"]
            row = {"model": m, "N": n, "attempted": len(rs), "measured": len(ms),
                   "nonconverged": sum(r["status"] == "NONCONVERGED" for r in rs),
                   "failed": sum(r["status"] == "FAILED" for r in rs)}
            row["converged_rate"] = len(ms) / len(rs) if rs else np.nan
            if ms:
                row["curve_rmse"] = float(np.mean([r["curve_rmse"] for r in ms]))
                row["mapping_maxdiff"] = float(max(r["mapping_maxdiff"] for r in ms))
                row["info_rel_maxdiff"] = float(max(r["info_rel_maxdiff"] for r in ms))
                row["eap_corr"] = float(np.mean([r["eap_corr"] for r in ms]))
                row["eap_rmse"] = float(np.mean([r["eap_rmse"] for r in ms]))
                for b in ms[0]["blocks"]:
                    row[f"rmse_{b}"] = float(np.mean([r["blocks"][b]["rmse"] for r in ms]))
                    row[f"bias_{b}"] = float(np.mean([r["blocks"][b]["bias"] for r in ms]))
                    cov = sum(r["blocks"][b]["covered"] for r in ms); nci = sum(r["blocks"][b]["n_ci"] for r in ms)
                    tot = sum(r["blocks"][b]["n_param"] for r in ms)
                    row[f"coverage_{b}"] = cov / nci if nci else np.nan
                    row[f"coverage_{b}_ci"] = cp(cov, nci) if nci else None
                    row[f"se_available_{b}"] = nci / tot if tot else np.nan
                ch = [r["choice"] for r in ms if "choice" in r]
                if ch:
                    row["aic_choice_true_rate"] = float(np.mean([c["aic_true"] for c in ch]))
                    row["bic_choice_true_rate"] = float(np.mean([c["bic_true"] for c in ch]))
                covs = [row[k] for k in row if k.startswith("coverage_") and not k.endswith("_ci") and np.isfinite(row[k])]
                row["meets_criteria"] = bool(row["converged_rate"] >= CRITERIA["converged_rate"]
                                             and row["info_rel_maxdiff"] <= CRITERIA["info_rel"]
                                             and row["eap_corr"] >= CRITERIA["eap_corr"]
                                             and (n != 2000 or (row["curve_rmse"] <= CRITERIA["curve_rmse"] and
                                                                all(CRITERIA["coverage"][0] <= c <= CRITERIA["coverage"][1] for c in covs))))
            out.append(row)
    return out


def main(outdir, quick=False):
    os.makedirs(outdir, exist_ok=True)
    js = jobs(quick); t0 = time.time()
    with Pool(int(os.environ.get("P4_PROCS", "2"))) as pool:
        recs = pool.map(fit_job, js, chunksize=1)
    payload = {"design": __doc__, "engine_path": ENGINE, "criteria": CRITERIA, "n_jobs": len(js), "wall_seconds": time.time() - t0,
               "versions": {"mirt": __import__("importlib.metadata").metadata.version("mirt")},
               "summary": summarize(recs), "runs": recs}
    name = "polytomous_mc_study.json" if ENGINE == "library_default" else f"polytomous_mc_study_{ENGINE}.json"
    with open(os.path.join(outdir, name), "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, default=float)
    return payload


if __name__ == "__main__":
    main(sys.argv[1], quick="--quick" in sys.argv)
