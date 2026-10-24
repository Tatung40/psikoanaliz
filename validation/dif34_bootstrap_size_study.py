"""Pre-registered size/power study of the 3PL/4PL parametric-bootstrap DIF LR test (warp-speed method).

Warp-speed Monte Carlo (Giacomini, Politis & White, 2013): each replication r draws ONE bootstrap
sample from its own fitted null model; the pooled bootstrap statistics {T*_r} estimate the bootstrap
null distribution and replication r rejects if T_r >= the 95 % quantile of {T*_r}. This evaluates the
size of the bootstrap test at the cost of 2 x 2 fits per replication.

DESIGN (fixed before results):
  generator : independent NumPy 3PL/4PL; k = 8; reference N = 500 ~ N(0,1); focal N = 500 ~ N(-.3, 1.1^2);
              a = linspace(.9, 1.7), b = linspace(-1, 1), c = (0, .02, .10, .15, .20, .05, .12, .18),
              d = linspace(.90, .97) (4PL). Studied item M1 (c = .02, near the bound), anchors = all others.
  conditions: 3PL-null (R = 400), 3PL-null with studied item M3 (c = .15, interior; R = 400),
              4PL-null (R = 300), 3PL-DIF with b_focal(M1) = b + .6 (R = 200, power).
  reported  : bootstrap rejection rate at .05 with Clopper-Pearson interval; naive chi-square
              (df 3 / 4) rejection rate; fraction of replications with a boundary estimate.
  criterion : the bootstrap p-value may be shown in the UI for a model only if, in every null condition of
              that model, the rejection-rate 95 % CI contains .05 and the point estimate is <= .075.
Run: PYTHONPATH=. python validation/dif34_bootstrap_size_study.py OUTDIR [--quick]
"""
import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.special import expit
from scipy.stats import beta as B, chi2

K = 8
A = np.linspace(.9, 1.7, K); BB = np.linspace(-1, 1, K)
C = np.array([0., .02, .10, .15, .20, .05, .12, .18]); D = np.linspace(.90, .97, K)


def generate(model, item, dif, seed):
    rng = np.random.default_rng(seed)
    g = np.r_[np.zeros(500, int), np.ones(500, int)]
    th = np.r_[rng.standard_normal(500), -.3 + 1.1 * rng.standard_normal(500)]
    b = np.tile(BB, (1000, 1))
    if dif:
        b[g == 1, item] += .6
    d = D if model == "4PL" else np.ones(K)
    x = (rng.random((1000, K)) < C + (d - C) * expit(A * (th[:, None] - b))).astype(float)
    return x, g


def job(args):
    from core.irt_dif34_bootstrap import _lr, _simulate
    from core.irt_mg34 import default_bounds
    cond, model, item, dif, seed = args
    rec = {"condition": cond, "model": model, "seed": seed, "status": "FAILED"}
    try:
        x, g = generate(model, item, dif, seed); obs = np.ones_like(x, bool)
        anchors = [j for j in range(K) if j != item]
        T, ok, f0, f1 = _lr(x, obs, g, model, item, anchors, default_bounds(), 41)
        if not ok:
            rec["status"] = "INVALID_OBSERVED"; return rec
        xs = _simulate(f0, g, obs, np.random.default_rng(seed + 7))
        Ts, oks, _, _ = _lr(xs, obs, g, model, item, anchors, default_bounds(), 41, f0["params"], f1["params"])
        rec.update(T=float(T), T_star=float(Ts), star_ok=bool(oks), boundary=bool(f0["at_bound"].any() or f1["at_bound"].any()),
                   naive_p=float(chi2.sf(T, 3 if model == "3PL" else 4)), status="OK")
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return rec


def jobs(quick=False):
    s = (lambda n: 10 if quick else n)
    out = [("3PL-null-M1", "3PL", 1, False, 500_000 + r) for r in range(s(400))]
    out += [("3PL-null-M3", "3PL", 3, False, 600_000 + r) for r in range(s(400))]
    out += [("4PL-null-M1", "4PL", 1, False, 700_000 + r) for r in range(s(300))]
    out += [("3PL-DIF-M1", "3PL", 1, True, 800_000 + r) for r in range(s(200))]
    return out


def cp(x, n):
    return [float(B.ppf(.025, x, n - x + 1)) if x else 0., float(B.ppf(.975, x + 1, n - x)) if x < n else 1.]


def summarize(recs):
    out = []
    for cond in sorted({r["condition"] for r in recs}):
        rs = [r for r in recs if r["condition"] == cond]
        ok = [r for r in rs if r["status"] == "OK" and r["star_ok"]]
        row = {"condition": cond, "attempted": len(rs), "used": len(ok),
               "invalid_or_failed": len(rs) - len(ok)}
        if ok:
            crit = float(np.quantile([r["T_star"] for r in ok], .95))
            rej = int(sum(r["T"] >= crit for r in ok)); naive = int(sum(r["naive_p"] < .05 for r in ok))
            row.update(bootstrap_critical_value=crit, bootstrap_rejection=rej / len(ok), bootstrap_ci=cp(rej, len(ok)),
                       naive_chi2_rejection=naive / len(ok), naive_ci=cp(naive, len(ok)),
                       boundary_fraction=float(np.mean([r["boundary"] for r in ok])))
            if "null" in cond:
                row["meets_criterion"] = bool(row["bootstrap_ci"][0] <= .05 <= row["bootstrap_ci"][1]
                                              and row["bootstrap_rejection"] <= .075)
        out.append(row)
    return out


def main(outdir, quick=False):
    os.makedirs(outdir, exist_ok=True)
    js = jobs(quick); t0 = time.time()
    with Pool(int(os.environ.get("P2_PROCS", "2"))) as pool:
        recs = pool.map(job, js, chunksize=1)
    summ = summarize(recs)
    status = {m: all(r.get("meets_criterion", False) for r in summ if r["condition"].startswith(m) and "null" in r["condition"])
              for m in ("3PL", "4PL")}
    payload = {"design": __doc__, "n_jobs": len(js), "wall_seconds": time.time() - t0, "summary": summ,
               "ui_enabled": status, "runs": recs}
    with open(os.path.join(outdir, "dif34_bootstrap_size_study.json"), "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, default=float)
    return payload


if __name__ == "__main__":
    main(sys.argv[1], quick="--quick" in sys.argv)
