"""P3 Fallback A: fitted-null parametric bootstrap of the UNCHANGED production S-X2 with full cold native refits.

Per dataset: native MAP fit, production S-X2 (mirt.itemfit, its own grouping/pooling). B=99 bootstrap datasets from
the fitted 3PL/4PL (theta~N(0,1), Bernoulli), each with a full cold native refit and a full recomputation of S-X2
including regrouping. p_j = (1 + #{S*_j >= S_j}) / (B+1). Failed/non-converged refits are not replaced; >1 failure
(>1%) or non-finite S_j makes the dataset non-evaluable (counted as rejection). Non-finite S*_j counts as exceedance.
"""
import argparse, json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import mirt
from scipy.special import expit
from validation.binary_certification.common import simulate, truth
from core.irt_python_mirt_adapter import fit_python_mirt

COLS = ["discrimination", "difficulty", "guessing", "upper"]; B = 99


def data(model, root, r, alt):
    seed = [root, int(model[0]), r, 0]; x = simulate(model, 3000, 12, seed).astype(np.uint8)
    if alt:
        p = truth(model, 12); rng = np.random.default_rng(seed + [1]); th = rng.standard_normal(3000)
        c, d = p["c"][0], (p["d"][0] if model == "4PL" else 1.0)
        x[:, 0] = rng.random(3000) < c + (d - c) / (1 + np.exp(-2 * p["a"][0] * (th - p["b"][0])))
    return x, seed


def fit_sx(x, model):
    f = fit_python_mirt(x, model=model, estimation="MAP", n_quadpts=61, max_iter=500, tol=1e-4).fit
    if not f.converged: return None, None
    return f, mirt.itemfit(f, x, statistics=["S_X2"])["S_X2"].to_numpy(float)


def one(job):
    model, root, r, alt = job; x, seed = data(model, root, r, alt); rec = {"model": model, "r": r, "seed": seed, "alt": alt}
    try:
        f, s = fit_sx(x, model)
        if f is None: rec["converged"] = False; return rec
        rec["converged"] = True; rec["sx"] = s.tolist()
        k = 4 if model == "4PL" else 3; e = f.coef()[COLS[:k]].to_numpy(float)
        a, b, c = e[:, 0], e[:, 1], e[:, 2]; d = e[:, 3] if k == 4 else np.ones(12)
        star = np.full((B, 12), np.nan); fails = 0
        for i in range(B):
            rng = np.random.default_rng([root, int(model[0]), r, i + 1, 7 if alt else 0]); th = rng.standard_normal(3000)
            xs = (rng.random((3000, 12)) < c + (d - c) * expit(a * (th[:, None] - b))).astype(np.uint8)
            try:
                fs, ss = fit_sx(xs, model)
                if fs is None: fails += 1
                else: star[i] = ss
            except Exception: fails += 1
        exceed = np.where(np.isfinite(star), star >= s, True).sum(0)
        p = (1 + exceed) / (B + 1)
        evaluable = np.isfinite(s) & (fails <= 1)
        rec.update(p_boot=np.where(evaluable, p, np.nan).tolist(), inner_failures=fails, star_mean=np.nanmean(star, 0).tolist())
    except Exception as ex:
        rec["error"] = repr(ex)
    return rec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model"); ap.add_argument("--root", type=int); ap.add_argument("--R", type=int)
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--step", type=int, default=1); ap.add_argument("--alt", action="store_true")
    ap.add_argument("--out"); ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args(); out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["r"] for l in out.read_text().splitlines()} if out.exists() else set()
    jobs = [(a.model, a.root, r, a.alt) for r in range(a.start, a.R, a.step) if r not in done]
    with ProcessPoolExecutor(a.workers) as ex, out.open("a") as fh:
        for rec in ex.map(one, jobs):
            fh.write(json.dumps(rec) + "\n"); fh.flush(); print("DONE", rec["model"], rec["r"], rec.get("inner_failures"), flush=True)


if __name__ == "__main__":
    sys.exit(main())
