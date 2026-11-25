"""Shared native-fit runner: P3 Method A p-values and P1 one-step candidates per dataset (JSONL, resumable)."""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import mirt
from scipy.stats import norm
from validation.binary_certification.common import simulate, truth
from core.irt_python_mirt_adapter import fit_python_mirt
from validation.p3_sx2_analytic import kernel as K

COLS = ["discrimination", "difficulty", "guessing", "upper"]


def one(job):
    model, root, r, alt = job
    seed = [root, int(model[0]), r, 0]
    x = simulate(model, 3000, 12, seed).astype(np.uint8)
    if alt:                                     # prespecified misfit: item 1 responses from a 2x steeper IRF
        p = truth(model, 12); rng = np.random.default_rng(seed + [1]); th = rng.standard_normal(3000)
        c, d = p["c"][0], (p["d"][0] if model == "4PL" else 1.0)
        x[:, 0] = rng.random(3000) < c + (d - c) / (1 + np.exp(-2 * p["a"][0] * (th - p["b"][0])))
    rec = {"model": model, "r": r, "seed": seed, "alt": alt}
    try:
        f = fit_python_mirt(x, model=model, estimation="MAP", n_quadpts=61, max_iter=500, tol=1e-4).fit
        rec["converged"] = bool(f.converged)
        if not f.converged: return rec
        k = 4 if model == "4PL" else 3
        eta = f.coef()[COLS[:k]].to_numpy(float).ravel()
        tab = mirt.itemfit(f, x, statistics=["S_X2"])
        res = K.analyse(x, eta, model)
        rec.update(sx_native=tab.S_X2.tolist(), p_chi=tab.p_value.tolist(), df=tab.df.tolist(),
                   sx=[q["sx"] for q in res], p_A=[q["p_A"] for q in res], p_A_nobias=[q["p_A_nobias"] for q in res],
                   sparse=[q["sparse"] for q in res], ngroups=[len(q["groups"]) for q in res],
                   groups=[q["groups"] for q in res], lam_sum=[float(np.sum(q["lam"])) for q in res])
        if not alt:                             # P1 one-step candidate (development only)
            px, score = K.pattern_terms(eta, model); I = (px[:, None] * score).T @ score
            gp, hp = K.log_prior_grad_hess(eta, model); A = I - np.diag(hp) / 3000
            Ii, Ai = np.linalg.inv(I), np.linalg.inv(A)
            tr = truth(model, 12); tv = np.stack([tr[q] for q in "abcd"[:k]], 1).ravel()
            one_step = eta - Ii @ gp / 3000
            rec.update(eta=eta.tolist(), truth=tv.tolist(), one_step=one_step.tolist(),
                       se_ml=np.sqrt(np.diag(Ii) / 3000).tolist(), se_map=np.sqrt(np.diag(Ai @ I @ Ai) / 3000).tolist())
    except Exception as e:
        rec["error"] = repr(e)
    return rec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--root", type=int, required=True)
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--R", type=int, required=True); ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--alt", action="store_true"); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(); out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["r"] for l in out.read_text().splitlines()} if out.exists() else set()
    jobs = [(a.model, a.root, r, a.alt) for r in range(a.start, a.R, a.step) if r not in done]
    with ProcessPoolExecutor(a.workers) as ex, out.open("a") as fh:
        for rec in ex.map(one, jobs, chunksize=2):
            fh.write(json.dumps(rec) + "\n"); fh.flush()


if __name__ == "__main__":
    sys.exit(main())
